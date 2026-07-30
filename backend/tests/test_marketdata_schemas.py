"""Tests for the values the market schemas *derive*.

Pass-through fields are not worth asserting -- pydantic already guarantees
those. What is worth asserting is the arithmetic, because it is done once here
so that no client has to, and a sign error would be reported identically by
every one of them.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from app.marketdata.domain import Earnings, MarketSession, Quote
from app.schemas.marketdata import EarningsResponse, QuoteResponse


def _quote(price: Decimal | None, previous: Decimal | None) -> Quote:
    return Quote(
        symbol="MU",
        timestamp=datetime(2026, 7, 30, tzinfo=UTC),
        price=price,
        previous_close=previous,
        session=MarketSession.REGULAR,
    )


class TestQuoteChange:
    """The move from the previous close."""

    def test_a_rise_is_positive(self) -> None:
        response = QuoteResponse.from_domain(_quote(Decimal("110"), Decimal("100")))

        assert response.change == Decimal("10")
        assert response.change_percent == pytest.approx(10.0)

    def test_a_fall_is_negative(self) -> None:
        response = QuoteResponse.from_domain(_quote(Decimal("90"), Decimal("100")))

        assert response.change == Decimal("-10")
        assert response.change_percent == pytest.approx(-10.0)

    @pytest.mark.parametrize(
        ("price", "previous"),
        [(Decimal("110"), None), (None, Decimal("100")), (Decimal("110"), Decimal("0"))],
        ids=["no-previous-close", "no-price", "zero-previous-close"],
    )
    def test_an_incomputable_change_is_absent_not_zero(
        self, price: Decimal | None, previous: Decimal | None
    ) -> None:
        """Zero would render as "unchanged", which is a different claim.

        A halted issue with no last trade has an *unknown* change, and a client
        shown 0.00% believes the stock is flat.
        """
        response = QuoteResponse.from_domain(_quote(price, previous))

        assert response.change is None
        assert response.change_percent is None

    def test_the_change_is_exact(self) -> None:
        """Decimal end to end: the subtraction must not acquire a binary error.

        In float arithmetic 874.66 - 739.00 is 135.66000000000008, and that
        trailing noise reaches the screen.
        """
        response = QuoteResponse.from_domain(_quote(Decimal("874.66"), Decimal("739.00")))

        assert response.change == Decimal("135.66")


def _earnings(actual: Decimal | None, estimate: Decimal | None) -> Earnings:
    return Earnings(
        symbol="MU",
        fiscal_date=date(2026, 6, 30),
        eps_actual=actual,
        eps_estimate=estimate,
    )


class TestEarningsSurprise:
    """A beat is positive; a miss is negative. Including at a loss."""

    def test_a_beat_is_positive(self) -> None:
        response = EarningsResponse.from_domain(_earnings(Decimal("2.20"), Decimal("2.00")))

        assert response.surprise_percent == pytest.approx(10.0)

    def test_a_miss_is_negative(self) -> None:
        response = EarningsResponse.from_domain(_earnings(Decimal("1.80"), Decimal("2.00")))

        assert response.surprise_percent == pytest.approx(-10.0)

    def test_a_smaller_loss_than_expected_is_a_beat(self) -> None:
        """The reason the denominator is ``abs()``.

        Expected -2.00, reported -1.00: the company lost half what analysts
        feared, which is unambiguously good news. Dividing by the signed
        estimate flips the sign and reports a 50% miss -- the single most
        misleading number this schema could produce, and one that only appears
        for loss-making companies.
        """
        response = EarningsResponse.from_domain(_earnings(Decimal("-1.00"), Decimal("-2.00")))

        assert response.surprise_percent == pytest.approx(50.0)

    def test_a_deeper_loss_than_expected_is_a_miss(self) -> None:
        response = EarningsResponse.from_domain(_earnings(Decimal("-3.00"), Decimal("-2.00")))

        assert response.surprise_percent == pytest.approx(-50.0)

    @pytest.mark.parametrize(
        ("actual", "estimate"),
        [(Decimal("2.0"), None), (None, Decimal("2.0")), (Decimal("2.0"), Decimal("0"))],
        ids=["no-estimate", "not-yet-reported", "zero-estimate"],
    )
    def test_an_incomputable_surprise_is_absent(
        self, actual: Decimal | None, estimate: Decimal | None
    ) -> None:
        """A quarter with no estimate has no surprise -- not a surprise of zero."""
        response = EarningsResponse.from_domain(_earnings(actual, estimate))

        assert response.surprise_percent is None
