"""Tests for the article tagger.

Precision matters more than recall here: a false positive attaches an article to
the wrong ticker, and the correlation engine then offers it as the explanation
for a real price move. Several of these tests exist purely to pin down cases
where a naive substring match would be wrong.
"""

from __future__ import annotations

import pytest

from app.models.enums import EcosystemTag
from app.services.tagging import ArticleTagger


@pytest.fixture
def tagger() -> ArticleTagger:
    """Tagger wired with the symbol mapping the seed data produces."""
    return ArticleTagger(
        symbols_by_slug={
            "micron": ["MU"],
            "nvidia": ["NVDA"],
            "amd": ["AMD"],
            "tsmc": ["TSM"],
            "sk-hynix": ["000660.KS"],
        }
    )


def test_company_mention_resolves_to_its_ticker(tagger: ArticleTagger) -> None:
    result = tagger.tag("Micron raised its HBM revenue outlook for the year.")

    assert result.company_slugs == ("micron",)
    assert result.tickers == ("MU",)
    assert result.is_relevant


def test_multiple_companies_are_all_captured(tagger: ArticleTagger) -> None:
    result = tagger.tag("TSMC will expand CoWoS packaging, easing supply of NVIDIA accelerators.")

    assert set(result.company_slugs) == {"tsmc", "nvidia"}
    assert set(result.tickers) == {"TSM", "NVDA"}


def test_segments_are_detected_from_domain_vocabulary(tagger: ArticleTagger) -> None:
    result = tagger.tag("High-bandwidth memory pricing firmed as DDR5 contracts settled.")

    assert EcosystemTag.HBM.value in result.tags
    assert EcosystemTag.DRAM.value in result.tags


def test_word_boundaries_prevent_false_positives(tagger: ArticleTagger) -> None:
    """Substrings must not fire: `amd` in `Amdahl`, `aws` in `laws`."""
    result = tagger.tag("Amdahl's law still constrains parallel speedups under new laws.")

    assert "amd" not in result.company_slugs
    assert EcosystemTag.HYPERSCALER.value not in result.tags


def test_matching_is_case_insensitive(tagger: ArticleTagger) -> None:
    lower = tagger.tag("nvidia announced a new gpu")
    upper = tagger.tag("NVIDIA ANNOUNCED A NEW GPU")

    assert lower.company_slugs == upper.company_slugs == ("nvidia",)
    assert lower.tags == upper.tags


def test_unrelated_article_is_marked_irrelevant(tagger: ArticleTagger) -> None:
    """Storing these would inflate the retrieval index with uncitable text."""
    result = tagger.tag("Ten houseplants that thrive in low light, and how to water them.")

    assert not result.is_relevant
    assert result.company_slugs == ()
    assert result.tags == ()


def test_a_segment_without_a_company_is_still_relevant(tagger: ArticleTagger) -> None:
    """Industry news with no named company still informs the HBM tracker."""
    result = tagger.tag("Contract DRAM prices rose 12% quarter over quarter.")

    assert result.is_relevant
    assert result.company_slugs == ()
    assert EcosystemTag.DRAM.value in result.tags


def test_output_is_deterministic(tagger: ArticleTagger) -> None:
    """Equal input must produce byte-identical documents.

    Non-deterministic ordering would make a re-ingested article look modified,
    defeating the deduplication the pipeline depends on.
    """
    text = "AMD and NVIDIA both rely on TSMC advanced packaging for HBM-equipped GPUs."

    first = tagger.tag(text)
    second = tagger.tag(text)

    assert first == second
    assert list(first.company_slugs) == sorted(first.company_slugs)


def test_unknown_company_produces_no_ticker() -> None:
    """A company with no listing in the mapping tags the company but no symbol."""
    tagger = ArticleTagger(symbols_by_slug={})

    result = tagger.tag("Micron reported record HBM revenue.")

    assert result.company_slugs == ("micron",)
    assert result.tickers == ()


def test_keywords_are_capped(tagger: ArticleTagger) -> None:
    """A long article must not attach an unbounded keyword list to its document."""
    text = " ".join(
        ["hbm dram nand gpu cpu foundry euv cowos chiplet interconnect server capex"] * 20
    )

    result = tagger.tag(text)

    assert len(result.keywords) <= 20


def test_product_codenames_resolve_to_their_vendor(tagger: ArticleTagger) -> None:
    """Analysts write "Blackwell", not "NVIDIA's Blackwell architecture"."""
    result = tagger.tag("Blackwell shipments accelerated through the quarter.")

    assert result.company_slugs == ("nvidia",)
