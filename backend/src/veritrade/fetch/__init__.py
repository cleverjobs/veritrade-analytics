"""Article fetching: SSRF guard, browser-like client, block/consent/paywall detection, single-flight cache."""

from veritrade.fetch.cache import CacheResult, SingleFlightCache, normalize_url
from veritrade.fetch.client import HttpFetcher
from veritrade.fetch.ssrf import check_target
from veritrade.fetch.types import Fetcher, FetchResult

__all__ = [
    "CacheResult",
    "FetchResult",
    "Fetcher",
    "HttpFetcher",
    "SingleFlightCache",
    "check_target",
    "normalize_url",
]
