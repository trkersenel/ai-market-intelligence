"""Vendor adapters.

Each translates one provider's wire format into the platform's domain types and
declares which capabilities it actually serves. Nothing outside this package
imports a concrete adapter -- callers resolve providers through the registry.
"""

from app.marketdata.providers.finnhub import FinnhubProvider

__all__ = ["FinnhubProvider"]
