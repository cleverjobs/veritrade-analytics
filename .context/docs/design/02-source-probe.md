# 02 - Source probe

Question (Q18, Q19): which of the 14 candidate publishers can a plain Python client fetch, with no evasion,
and what do the extractors need to know about each one? The answer is measured, not assumed:
`scripts/probe_sources.py` fetched every candidate live on 2026-10-18 and its output decided
`backend/src/veritrade/data/sources.toml`.

## Method

- **Client**: `httpx 0.28.1` `AsyncClient`, HTTP/2, the Chrome 155 desktop header set from
  `veritrade.fetch.headers` (UA, Accept, `Accept-Language: en-US`, client hints, `Sec-Fetch-*`,
  `Upgrade-Insecure-Requests`, `Priority`), `Accept-Encoding: gzip, deflate, br` because `brotli` is installed.
  `sec.gov` gets the declared `SEC_USER_AGENT` and no Chrome hints. No cookies are pre-seeded, no TLS
  impersonation, headless browser, proxy or captcha solving (Q19).
- **Redirects**: followed by hand (max 5), exactly as the serving fetcher will; a hop to Yahoo's consent hosts is
  answered once with reject-all (`veritrade.fetch.consent`) and the original URL retried.
- **Politeness**: one request at a time per registrable domain with at least 1 s between them; EDGAR at 5 req/s
  (half its 10 req/s fair-access ceiling). Reported fetch times exclude these waits.
- **Discovery**: each source's `feeds` (RSS/Atom, non-English `dc:language` items skipped) or `sections`
  (HTML pages and sitemaps), filtered by `article_pattern`; the 3 newest matches are probed. EDGAR resolves the
  8-K Atom feed to filing indexes and then to the EX-99.1 exhibit, preferring index descriptions that say press
  release, earnings or results. When a source refuses its own hubs, links to it are harvested from the other
  sources' captured pages (AP).
- **Per article** (all in `evals/data/source_probe.json`, metadata only): status, final URL, redirects, bytes,
  network ms, `detect.classify` outcome and time, JSON-LD types / `articleBody` words / `isAccessibleForFree` /
  `hasPart` selectors, framework JSON markers, ranked body-container candidates with paragraph and word counts
  under both parsers, noise blocks inside the best container (selectors only) and press-release boilerplate
  labels (label and position only). Raw HTML goes to `evals/data/private/probe/<source>/<n>.html` (gitignored).
- **Profile check**: every `[sources.<key>.profile]` is applied with selectolax (Lexbor) and lxml `cssselect`;
  it passes when both parsers keep the same paragraphs in the same order (non-whitespace characters within 1%)
  and the body has at least 120 words. `--reanalyze` reruns all analysis on the saved HTML without network,
  which is how the profiles were tuned.

## Results

Vantage: residential ISP in Bulgaria (EU), macOS, Sunday 2026-10-18. Fetch ms is network time per article
(median, max of 3); size is the median HTML size.

| Source | Discovery used | OK | Fetch ms | Size | JSON-LD NewsArticle / articleBody | `isAccessibleForFree` | Blocks, consent | Profile |
|---|---|---|---|---|---|---|---|---|
| Yahoo Finance | `/topic/stock-market-news/` page (RSS now 404) | 3/3 | 477 (575) | 682 KB | 3 / 0 | absent | none; no consent redirect | 3/3 |
| CNBC | RSS `id/100003114` | 3/3 | 161 (221) | 838 KB | 3 / 0 | absent | none | 3/3 |
| Motley Fool | RSS `foolwatch` | 3/3 | 202 (235) | 372 KB | 3 / 0 | absent | none | 3/3 |
| PR Newswire | RSS financial services | 3/3 | 34 (43) | 219 KB | 3 / 0 | absent | none | 3/3 |
| GlobeNewswire | RSS public companies | 3/3 | 552 (662) | 74 KB | 3 / 0 | true | none | 3/3 |
| SEC 8-K EX-99.1 | Atom, 25 indexes, 2 releases + 1 statement | 3/3 | 135 (538) | 19 KB | 0 / 0 | absent | none (declared UA) | 3/3 |
| CNN Business | `edition.cnn.com/business` page (no RSS) | 3/3 | 35 (50) | 5.7 MB | 3 / 3 | false, full text served | none; www redirects to edition for EU | 3/3 |
| Fortune | RSS | 3/3 | 367 (495) | 431 KB | 3 / 0 | "False", full text served | none | 3/3 |
| Guardian Business | RSS `uk/business` | 3/3 | 116 (169) | 412 KB | 3 / 0 | true | none | 3/3 |
| BBC Business | RSS | 3/3 | 26 (62) | 394 KB | 3 / 0 | absent | none | 3/3 |
| AP News | hub + news sitemap both 403; 3 links harvested | 0/3 | - | - | - | - | Cloudflare managed challenge (403, `cf-mitigated: challenge`) on every path | - |
| Investing.com | RSS works | 0/3 | - | - | - | - | Cloudflare 403 with a 3-byte body on every HTML page | - |
| Benzinga | Google News sitemap works | 0/3 | - | - | - | - | Cloudflare "Sorry, you have been blocked" 403 on every HTML page | - |
| Business Wire | MRSS feed works | 0/3 | - | - | - | - | Akamai 403 "Page Unavailable" on releases; `/newsroom` serves an Akamai interactive challenge with HTTP 200 | - |

Every response came over HTTP/2 and no article URL redirected. One PR Newswire release took 6.8 s on a cache
miss during an earlier run, close to the 8 s fetch deadline.

## Decision

Final 10 (`status = "primary"`, `rank` in `sources.toml`), plan order with backups swapped in for the
rejected sources:

1. Yahoo Finance
2. CNBC
3. The Motley Fool
4. PR Newswire
5. GlobeNewswire
6. SEC EDGAR 8-K EX-99.1
7. CNN Business
8. Fortune (replaces AP News)
9. The Guardian Business (replaces Investing.com)
10. BBC Business (replaces Benzinga)

The mix is 7 news outlets, 2 press-release wires and 1 regulatory filing source. **No backups remain**: the
fourth backup, Business Wire, is blocked too. If a primary degrades, rerun the probe with `--only` on the
candidates and promote from there.

Rejected (`status = "rejected"`, `reason` in `sources.toml`): AP News, Investing.com, Benzinga, Business Wire,
all refused at the CDN (Cloudflare or Akamai) before any content is served. These outlets are syndicated
widely, so `detect.py` answers a block with a hint that points at the syndicated copy (usually Yahoo Finance).

## Yahoo EU consent

- From this IP, `finance.yahoo.com`, `yahoo.com/news`, `uk.finance.yahoo.com` and `de.yahoo.com` served content
  directly with no consent redirect. The pages carry `"gdpr":true` and a client-side CMP
  (`consent.cmp.oath.com`), so in October 2026 the server-side wall no longer stands in front of articles.
- The server-side flow still exists at `guce.yahoo.com/consent` -> `consent.yahoo.com/v2/collectConsent`, so the
  probe exercises it on every run (`yahoo_consent_check` in the JSON): `consent.reject_all` posted the
  reject-all form in 857 ms and the jar received `A1`, `A1S`, `CFC`, `EuConsent`, `GUC`. Decoding the TCF v2
  string in `EuConsent` gives **0 of 24 purposes consented**, which confirms that reject-all was recorded and
  accept-all was never sent. A second pass with the same jar skips the form (about 120 ms, no POST).
- Serving contract: when a redirect lands on a consent host, the fetcher calls `reject_all` with its SSRF
  guard and retries the original URL once; if that lands on the wall again, `detect.classify` raises
  `consent_required` with a hint.

## Block, consent and paywall detection

`detect.classify(status, headers, html, final_url)` is pure and string-based, so it runs on every fetch.

| Signal | Raised as |
|---|---|
| Final URL on `consent.yahoo.com` / `guce.yahoo.com`, or a Yahoo consent form in the page | `consent_required` |
| HTTP 401, 403, 429 (retryable), 451 | `source_blocked`, naming the CDN edge from `server` / `cf-ray` |
| Cloudflare: `cf-mitigated: challenge`, "Just a moment...", `_cf_chl_opt`, `/cdn-cgi/challenge-platform/h/`, "Attention Required! \| Cloudflare" | `source_blocked` |
| DataDome `captcha-delivery.com` or `x-dd-b`; PerimeterX `px-captcha` or "Press & Hold"; Akamai "Access Denied" + "Reference #", `sec-if-cpt-container`, `errors.edgesuite.net`; Imperva `_Incapsula_Resource` | `source_blocked`, including when served with HTTP 200 |
| JSON-LD `isAccessibleForFree` false (bool or string) and both the JSON-LD `articleBody` and the visible `<p>` text under 200 words | `paywalled` |

Design points the probe forced:

- Cloudflare's passive `/cdn-cgi/challenge-platform/scripts/jsd/main.js` beacon sits on normal pages, so only
  the interactive `/h/` path counts. `x-datadome` alone is ignored because DataDome sets it on allowed responses.
- Markers are only searched in the first 64 KB: interstitials are a few KB, and it keeps 6 MB CNN pages cheap.
  Over all 42 captured pages `classify` takes at most 1.7 ms (CNN, 5.8 MB) and under 1 ms everywhere else
  (median of 5 runs).
- CNN and Fortune declare `isAccessibleForFree: false` yet serve the whole article (their walls are
  client-side), which is why a not-free flag alone is never treated as a paywall.
- Hints are keyed by host suffix: Reuters, AP, Business Wire and Benzinga blocks point to the syndicated copy on
  Yahoo Finance; WSJ, Barron's, FT and Bloomberg paywalls point to free sources.

## What the extractors get

Profile semantics (documented at the top of `sources.toml`): remove `drop` selectors, take `body` matches in
document order, ignore paragraphs under `min_words`, then cut at the first paragraph after the lede that starts
with a `stop_markers` prefix. All selectors use only tag, class, id, attribute and descendant parts; GlobeNewswire
also uses a selector list for `p, li`. All 30 primary pages pass the two-parser check.

Findings that are not visible in the selectors:

- **JSON-LD `articleBody` exists only on CNN** (3/3). Everywhere else the cascade has to fall through to the
  profile or density strategy. No source ships `__NEXT_DATA__`; Fool and Fortune use the Next.js flight stream
  and Yahoo SvelteKit, neither worth parsing for the body.
- **Text joining differs between parsers**: selectolax `text(separator=" ")` splits words at inline tags (drop
  caps become "R ogue", "Fla. ,"), while lxml `text_content()` glues text across `<br>` ("WangEmail"). Joining
  text nodes without a separator, with spaces only at `<br>` and block boundaries, gives the same text on both.
- **Press releases** end with an "About <issuer>" heading, forward-looking statements and contacts as plain
  paragraphs, so they are cut with stop markers, not selectors. The "About" check only applies after the lede.
- **SEC exhibits** arrive wrapped in the EDGAR SGML envelope, with no `h1`, JSON-LD or containers, and start with
  "Exhibit 99.1" and the issuer name. Some filer agents emit `div`/`span` text without `<p>`, which this profile
  cannot read; the cascade must fall back to density there. EX-99.1 is sometimes a financial statement (11.6k
  words in one sample), which `scope=lede` and the token cap absorb.
- **Yahoo** stories are mostly partner copy (Motley Fool, Insider Monkey, 24/7 Wall St, named in `providerId`)
  and partner promos survive inside the body; the second half of the text sits in a hidden "Story continues"
  block that must be kept.
- **CNBC** has editor-written Key Points (`div.RenderKeyPoints-list li`) outside the body, a ready-made lede.
- **CNN** pages are 5.7 MB, so DOM parsing is the slow step there; the JSON-LD strategy avoids it.

## Caveats

- One vantage point. Bot management depends on IP reputation: GitHub Actions runners use US datacenter IPs that
  Cloudflare and Akamai treat more strictly, and US visitors see different geo behaviour (no Yahoo consent at
  all, `www.cnn.com`, `bbc.com`). CI therefore runs on skeleton fixtures; the weekly live smoke is informational.
- Three articles per source on a Sunday: the wires were dominated by non-US issuers, and only 2 of the 25 newest
  8-K filing indexes had an EX-99.1 described as a press release.
- Blocking decisions drift over time and with request rate; the probe stayed at or under 1 request per second per
  domain. The Business Wire `/newsroom` challenge was seen while exploring and is not in the final JSON, because
  discovery uses its MRSS feed.
- AP's article URLs were harvested from other publishers' pages because its own hubs and sitemaps are blocked.
- A Chrome TLS fingerprint might get past Cloudflare for AP, Benzinga and Investing.com. That is evasion, which
  Q19 rules out.

## Reproduce

```bash
uv run python scripts/probe_sources.py --vantage "EU (Bulgaria) residential ISP"   # live, about 2 minutes
uv run python scripts/probe_sources.py --only cnbc,bbc                               # re-probe some sources
uv run python scripts/probe_sources.py --reanalyze                                   # re-score saved HTML offline
```
