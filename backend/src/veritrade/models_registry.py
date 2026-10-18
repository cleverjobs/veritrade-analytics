"""Model allowlist and per-endpoint defaults loaded from `data/models.toml`."""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from functools import lru_cache
from importlib import resources
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from veritrade.config import Settings, get_settings
from veritrade.errors import ModelNotAllowed
from veritrade.inference.types import Endpoint, FieldOrder, ModelKind, PolicyName

ENDPOINTS: tuple[Endpoint, ...] = get_args(Endpoint)
SignalSchema = Literal["label", "ordinal"]


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    kind: ModelKind
    endpoints: tuple[Endpoint, ...] = Field(min_length=1)
    num_ctx: int | None = Field(None, gt=0)
    max_input_tokens: int = Field(gt=0)
    num_predict: dict[Endpoint, int] = Field(default_factory=dict)
    signal_schema: SignalSchema | None = None
    policy: PolicyName
    # 0 never abstains on the winning class.
    min_p: float | None = Field(None, ge=0, le=1)
    ordinal_t: int = Field(1, ge=1, le=2)
    directional_tau: float | None = Field(None, gt=0, lt=1)
    field_order: FieldOrder | None = None
    # Frozen prompt version per endpoint; an endpoint left out serves the prompt file's `active`.
    prompt: dict[Endpoint, str] = Field(default_factory=dict)
    evaluated: bool = True

    @model_validator(mode="before")
    @classmethod
    def _default_field_order(cls, data: Any) -> Any:
        if isinstance(data, dict) and "analyze" in data.get("endpoints", ()) and not data.get("field_order"):
            return {**data, "field_order": "sentiment-first"}
        return data

    @model_validator(mode="after")
    def _check_consistency(self) -> ModelSpec:
        if self.kind == "decision":
            if self.endpoints != ("signal",) or self.policy != "probability":
                raise ValueError(
                    f"{self.name}: decision models serve only /signal with the probability policy"
                )
        else:
            if self.policy == "probability":
                raise ValueError(f"{self.name}: generative models have no native class probabilities")
            if self.num_ctx is None:
                raise ValueError(f"{self.name}: generative models need num_ctx")
            missing = [e for e in self.endpoints if e not in self.num_predict]
            if missing:
                raise ValueError(f"{self.name}: num_predict missing for {missing}")
            if "signal" in self.endpoints and self.signal_schema is None:
                raise ValueError(f"{self.name}: generative /signal models need signal_schema")
            if "signal" in self.endpoints and self.policy == "ordinal" and self.signal_schema != "ordinal":
                raise ValueError(f"{self.name}: the ordinal policy needs signal_schema = 'ordinal'")
        if self.policy in ("probability", "label-logprob") and self.min_p is None:
            raise ValueError(f"{self.name}: policy {self.policy} needs min_p")
        if self.directional_tau is not None and self.policy == "ordinal":
            raise ValueError(
                f"{self.name}: directional_tau needs class probabilities, not the ordinal policy"
            )
        if stray := sorted(set(self.prompt) - set(self.endpoints)):
            raise ValueError(f"{self.name}: prompt versions for endpoints it does not serve: {stray}")
        return self

    def serves(self, endpoint: Endpoint) -> bool:
        return endpoint in self.endpoints


def parse_models_toml(text: str) -> tuple[dict[str, ModelSpec], dict[Endpoint, str]]:
    raw = tomllib.loads(text)
    specs = {name: ModelSpec(name=name, **body) for name, body in raw.get("models", {}).items()}
    defaults: dict[Endpoint, str] = {}
    for endpoint, name in raw.get("defaults", {}).items():
        if endpoint not in ENDPOINTS:
            raise ValueError(f"models.toml: unknown endpoint in [defaults]: {endpoint}")
        defaults[endpoint] = name
    return specs, defaults


def read_models_toml() -> str:
    return resources.files("veritrade.data").joinpath("models.toml").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def allowlist() -> Mapping[str, ModelSpec]:
    """Every model in models.toml, ignoring VT_MODELS (prompts and evals need the full roster)."""
    return parse_models_toml(read_models_toml())[0]


class Registry:
    def __init__(
        self,
        specs: Mapping[str, ModelSpec],
        defaults: Mapping[Endpoint, str],
        *,
        restriction: Iterable[str] = (),
        overrides: Mapping[Endpoint, str | None] | None = None,
    ) -> None:
        restricted = list(restriction)
        unknown = [name for name in restricted if name not in specs]
        if unknown:
            raise ValueError(f"VT_MODELS names models outside models.toml: {', '.join(unknown)}")
        keep = set(restricted) if restricted else set(specs)
        self._all = dict(specs)
        self._specs = {name: spec for name, spec in specs.items() if name in keep}
        self._defaults: dict[Endpoint, str] = {}
        for endpoint in ENDPOINTS:
            override = (overrides or {}).get(endpoint)
            if override:
                if not self._is_allowed(endpoint, override):
                    raise ValueError(f"default model override {override!r} is not allowed for /{endpoint}")
                self._defaults[endpoint] = override
                continue
            configured = defaults.get(endpoint)
            if configured and self._is_allowed(endpoint, configured):
                self._defaults[endpoint] = configured
            elif fallback := self.allowed(endpoint):
                self._defaults[endpoint] = fallback[0].name

    @classmethod
    def load(cls, settings: Settings | None = None, *, toml_text: str | None = None) -> Registry:
        settings = settings or get_settings()
        specs, defaults = parse_models_toml(toml_text if toml_text is not None else read_models_toml())
        overrides: dict[Endpoint, str | None] = {
            "signal": settings.default_model_signal,
            "summary": settings.default_model_summary,
            "analyze": settings.default_model_analyze,
        }
        return cls(specs, defaults, restriction=settings.model_restriction, overrides=overrides)

    def _is_allowed(self, endpoint: Endpoint, name: str) -> bool:
        spec = self._specs.get(name)
        return spec is not None and spec.serves(endpoint)

    @property
    def models(self) -> list[ModelSpec]:
        return list(self._specs.values())

    @property
    def defaults(self) -> dict[Endpoint, str]:
        return dict(self._defaults)

    def allowed(self, endpoint: Endpoint) -> list[ModelSpec]:
        return [spec for spec in self._specs.values() if spec.serves(endpoint)]

    def resolve(self, endpoint: Endpoint, model: str | None) -> ModelSpec:
        allowed = ", ".join(spec.name for spec in self.allowed(endpoint)) or "none"
        hint = f"Allowed for /{endpoint}: {allowed}."
        if model is None:
            default = self._defaults.get(endpoint)
            if default is None:
                raise ModelNotAllowed(f"No model is configured for /{endpoint}.", hint=hint)
            return self._specs[default]
        spec = self._specs.get(model)
        if spec is None:
            reason = "is disabled by VT_MODELS" if model in self._all else "is not on the allowlist"
            raise ModelNotAllowed(f"Model {model!r} {reason}.", hint=hint)
        if not spec.serves(endpoint):
            raise ModelNotAllowed(f"Model {model!r} does not serve /{endpoint}.", hint=hint)
        return spec


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    return Registry.load()
