import pytest

from veritrade.config import Settings
from veritrade.errors import ERROR_STATUS, ErrorCode, VtError


def test_every_error_code_has_a_status() -> None:
    assert set(ERROR_STATUS) == set(ErrorCode)


def test_every_error_subclass_maps_to_its_status() -> None:
    for cls in VtError.__subclasses__():
        assert cls("x").http_status == ERROR_STATUS[cls.code]


def test_ssrf_test_hosts_refused_outside_test_env() -> None:
    with pytest.raises(ValueError, match="VT_ENV=test"):
        Settings(env="prod", ssrf_test_allow_hosts="fixtures.local")
    assert Settings(env="test", ssrf_test_allow_hosts="Fixtures.Local").ssrf_allow_hosts == {"fixtures.local"}
