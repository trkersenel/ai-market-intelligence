"""Resolve free-text articles to companies, tickers and ecosystem segments.

This is what turns a stream of headlines into evidence the correlation engine
can use: without tags, "why did Micron move?" cannot retrieve the article that
explains it, because nothing links the prose to the ticker.

Deliberately lexical, not learned. A rules-based tagger is inspectable -- when a
tag is wrong you can see exactly which alias fired -- and it has no cold-start
problem. The semantic layer arrives with embeddings; it complements this rather
than replacing it, because an article that never names Micron can still be
retrieved by meaning, and one that names it should never be missed by chance.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from app.models.enums import EcosystemTag

#: Aliases that identify a company in prose, beyond its canonical name.
#: Deliberately conservative: a false positive attaches an article to the wrong
#: ticker and becomes a wrong explanation for a real price move.
COMPANY_ALIASES: dict[str, tuple[str, ...]] = {
    "nvidia": ("nvidia", "nvda", "jensen huang", "blackwell", "hopper gpu"),
    "amd": ("amd", "advanced micro devices", "instinct mi300", "instinct mi325", "lisa su"),
    "micron": ("micron", "micron technology"),
    "sk-hynix": ("sk hynix", "sk-hynix", "hynix"),
    "samsung-electronics": ("samsung electronics", "samsung semiconductor"),
    "tsmc": ("tsmc", "taiwan semiconductor", "cowos"),
    "broadcom": ("broadcom", "avgo", "tomahawk switch"),
    "intel": ("intel", "intel foundry", "gaudi accelerator"),
    "asml": ("asml", "euv lithography", "high-na euv"),
    "super-micro": ("supermicro", "super micro", "smci"),
    "ge-vernova": ("ge vernova", "gevernova"),
}

#: Value-chain segments and the phrases that signal them.
TAG_KEYWORDS: dict[EcosystemTag, tuple[str, ...]] = {
    EcosystemTag.HBM: ("hbm", "high bandwidth memory", "high-bandwidth memory", "hbm3", "hbm4"),
    EcosystemTag.DRAM: ("dram", "ddr5", "lpddr", "memory pricing", "memory prices"),
    EcosystemTag.NAND: ("nand", "ssd", "flash memory"),
    EcosystemTag.GPU: ("gpu", "graphics processor", "accelerator", "ai chip", "ai chips"),
    EcosystemTag.CPU: ("cpu", "processor", "xeon", "epyc"),
    EcosystemTag.FOUNDRY: ("foundry", "fab", "wafer", "3nm", "2nm", "node"),
    EcosystemTag.LITHOGRAPHY: ("lithography", "euv", "duv", "scanner"),
    EcosystemTag.ADVANCED_PACKAGING: (
        "advanced packaging",
        "cowos",
        "chiplet",
        "2.5d packaging",
        "3d stacking",
        "interposer",
    ),
    EcosystemTag.NETWORKING: (
        "networking",
        "infiniband",
        "ethernet switch",
        "interconnect",
        "nvlink",
    ),
    EcosystemTag.SERVERS: ("server", "rack", "liquid cooling", "ai server"),
    EcosystemTag.HYPERSCALER: (
        "hyperscaler",
        "microsoft azure",
        "google cloud",
        "aws",
        "meta platforms",
        "openai",
        "anthropic",
        "capex",
    ),
    EcosystemTag.POWER_INFRASTRUCTURE: (
        "power grid",
        "turbine",
        "data center power",
        "electricity demand",
        "substation",
    ),
    EcosystemTag.EDA: ("eda", "cadence design", "synopsys"),
}


@dataclass(frozen=True)
class TagResult:
    """Everything the tagger extracted from one article."""

    company_slugs: tuple[str, ...] = field(default_factory=tuple)
    tickers: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    keywords: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_relevant(self) -> bool:
        """Whether the article relates to the tracked ecosystem at all.

        Feeds carry consumer reviews and unrelated tech news; an article with no
        company and no segment is noise that would otherwise be embedded, stored
        and retrieved at a cost with no analytical value.
        """
        return bool(self.company_slugs or self.tags)


class ArticleTagger:
    """Matches article text against company aliases and segment keywords."""

    def __init__(self, symbols_by_slug: Mapping[str, Sequence[str]] | None = None) -> None:
        """Build the tagger.

        Args:
            symbols_by_slug: Maps a company slug to its ticker symbols, so a
                match on "Micron" also tags ``MU``. Supplied from the database
                rather than hardcoded, so adding a listing needs no code change.
        """
        self._symbols_by_slug = {
            slug: tuple(symbols) for slug, symbols in (symbols_by_slug or {}).items()
        }
        self._company_patterns = {
            slug: _compile_alias_pattern(aliases) for slug, aliases in COMPANY_ALIASES.items()
        }
        self._tag_patterns = {
            tag: _compile_alias_pattern(keywords) for tag, keywords in TAG_KEYWORDS.items()
        }

    def tag(self, text: str) -> TagResult:
        """Extract companies, tickers, segments and keywords from ``text``.

        Args:
            text: Title, summary and body concatenated.

        Returns:
            The matches found. Order is deterministic -- companies by slug, tags
            by enum declaration order -- so equal inputs produce byte-identical
            documents and a re-ingested article does not look modified.
        """
        haystack = text.lower()

        slugs = tuple(
            slug
            for slug, pattern in sorted(self._company_patterns.items())
            if pattern.search(haystack)
        )
        tags = tuple(
            tag.value for tag, pattern in self._tag_patterns.items() if pattern.search(haystack)
        )
        keywords = tuple(
            keyword
            for tag, pattern in self._tag_patterns.items()
            for keyword in _matched_terms(pattern, haystack)
            if tag.value in tags
        )

        symbols = tuple(symbol for slug in slugs for symbol in self._symbols_by_slug.get(slug, ()))
        return TagResult(
            company_slugs=slugs,
            tickers=tuple(dict.fromkeys(symbols)),
            tags=tags,
            keywords=tuple(dict.fromkeys(keywords))[:20],
        )


def _compile_alias_pattern(aliases: Iterable[str]) -> re.Pattern[str]:
    r"""Compile aliases into one alternation with word boundaries.

    ``\b`` matters more than it looks: without it "amd" matches inside "amdahl"
    and "aws" inside "laws", quietly attributing unrelated articles to a ticker.
    Multi-word aliases are escaped so punctuation in them is literal.
    """
    escaped = sorted((re.escape(alias) for alias in aliases), key=len, reverse=True)
    return re.compile(rf"\b(?:{'|'.join(escaped)})\b", re.IGNORECASE)


def _matched_terms(pattern: re.Pattern[str], haystack: str) -> list[str]:
    """Return the distinct terms of ``pattern`` that occur in ``haystack``."""
    return list(dict.fromkeys(match.group(0).lower() for match in pattern.finditer(haystack)))
