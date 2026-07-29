"""The tracked universe: companies and listings the platform follows.

Kept as typed Python data rather than a SQL fixture so it is reviewable in a
diff, importable by tests, and validated by the same enums the ORM uses. This is
reference data, not user data -- it belongs in version control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models.enums import AssetType, EcosystemTag


@dataclass(frozen=True)
class TickerSeed:
    """A listing to track."""

    symbol: str
    display_name: str
    exchange: str
    currency: str = "USD"
    asset_type: AssetType = AssetType.EQUITY


@dataclass(frozen=True)
class CompanySeed:
    """A company and its listings."""

    slug: str
    name: str
    sector: str
    industry: str
    country: str
    website: str
    description: str
    tags: tuple[EcosystemTag, ...]
    tickers: tuple[TickerSeed, ...] = field(default_factory=tuple)


COMPANIES: tuple[CompanySeed, ...] = (
    CompanySeed(
        slug="nvidia",
        name="NVIDIA",
        sector="Information Technology",
        industry="Semiconductors",
        country="US",
        website="https://www.nvidia.com",
        description=(
            "Designs the GPUs and networking systems that most AI training and "
            "inference capacity is built on. The largest single consumer of "
            "high-bandwidth memory and of TSMC's advanced packaging capacity."
        ),
        tags=(EcosystemTag.GPU, EcosystemTag.NETWORKING, EcosystemTag.ADVANCED_PACKAGING),
        tickers=(TickerSeed("NVDA", "NVIDIA Corporation", "NASDAQ"),),
    ),
    CompanySeed(
        slug="amd",
        name="AMD",
        sector="Information Technology",
        industry="Semiconductors",
        country="US",
        website="https://www.amd.com",
        description=(
            "Supplies data-centre CPUs and the Instinct accelerator line, the "
            "principal merchant alternative to NVIDIA in AI training silicon."
        ),
        tags=(EcosystemTag.GPU, EcosystemTag.CPU, EcosystemTag.ADVANCED_PACKAGING),
        tickers=(TickerSeed("AMD", "Advanced Micro Devices, Inc.", "NASDAQ"),),
    ),
    CompanySeed(
        slug="micron",
        name="Micron Technology",
        sector="Information Technology",
        industry="Semiconductor Memory",
        country="US",
        website="https://www.micron.com",
        description=(
            "One of three suppliers of HBM at scale, alongside SK Hynix and "
            "Samsung. Its DRAM pricing and HBM sell-out commentary are the "
            "clearest public read on AI memory demand."
        ),
        tags=(EcosystemTag.HBM, EcosystemTag.DRAM, EcosystemTag.NAND),
        tickers=(TickerSeed("MU", "Micron Technology, Inc.", "NASDAQ"),),
    ),
    CompanySeed(
        slug="sk-hynix",
        name="SK Hynix",
        sector="Information Technology",
        industry="Semiconductor Memory",
        country="KR",
        website="https://www.skhynix.com",
        description=(
            "The leading HBM supplier by share, and NVIDIA's principal memory "
            "partner. Listed in Seoul, so its sessions lead the US market."
        ),
        tags=(EcosystemTag.HBM, EcosystemTag.DRAM, EcosystemTag.NAND),
        tickers=(TickerSeed("000660.KS", "SK hynix Inc.", "KRX", currency="KRW"),),
    ),
    CompanySeed(
        slug="samsung-electronics",
        name="Samsung Electronics",
        sector="Information Technology",
        industry="Semiconductor Memory",
        country="KR",
        website="https://www.samsung.com/semiconductor",
        description=(
            "Memory manufacturer and foundry. The third HBM supplier and the "
            "only credible leading-edge alternative to TSMC outside Intel."
        ),
        tags=(
            EcosystemTag.HBM,
            EcosystemTag.DRAM,
            EcosystemTag.NAND,
            EcosystemTag.FOUNDRY,
        ),
        tickers=(TickerSeed("005930.KS", "Samsung Electronics Co., Ltd.", "KRX", currency="KRW"),),
    ),
    CompanySeed(
        slug="tsmc",
        name="TSMC",
        sector="Information Technology",
        industry="Semiconductor Foundry",
        country="TW",
        website="https://www.tsmc.com",
        description=(
            "Manufactures essentially every leading-edge AI accelerator. Its "
            "CoWoS advanced-packaging capacity has been the binding constraint "
            "on industry-wide accelerator supply."
        ),
        tags=(EcosystemTag.FOUNDRY, EcosystemTag.ADVANCED_PACKAGING),
        tickers=(TickerSeed("TSM", "Taiwan Semiconductor Manufacturing Company (ADR)", "NYSE"),),
    ),
    CompanySeed(
        slug="broadcom",
        name="Broadcom",
        sector="Information Technology",
        industry="Semiconductors",
        country="US",
        website="https://www.broadcom.com",
        description=(
            "Co-designs the custom accelerators hyperscalers use to reduce "
            "dependence on merchant GPUs, and supplies the switching silicon "
            "that connects them."
        ),
        tags=(EcosystemTag.NETWORKING, EcosystemTag.ADVANCED_PACKAGING, EcosystemTag.GPU),
        tickers=(TickerSeed("AVGO", "Broadcom Inc.", "NASDAQ"),),
    ),
    CompanySeed(
        slug="intel",
        name="Intel",
        sector="Information Technology",
        industry="Semiconductors",
        country="US",
        website="https://www.intel.com",
        description=(
            "Data-centre CPU incumbent and an aspiring leading-edge foundry. "
            "Included as the counter-position to the TSMC-centric supply chain."
        ),
        tags=(EcosystemTag.CPU, EcosystemTag.FOUNDRY, EcosystemTag.ADVANCED_PACKAGING),
        tickers=(TickerSeed("INTC", "Intel Corporation", "NASDAQ"),),
    ),
    CompanySeed(
        slug="asml",
        name="ASML",
        sector="Information Technology",
        industry="Semiconductor Equipment",
        country="NL",
        website="https://www.asml.com",
        description=(
            "Sole supplier of EUV lithography. Its bookings lead foundry and "
            "memory capacity expansion by roughly two years, making it the "
            "earliest signal in the chain."
        ),
        tags=(EcosystemTag.LITHOGRAPHY, EcosystemTag.FOUNDRY),
        tickers=(TickerSeed("ASML", "ASML Holding N.V. (ADR)", "NASDAQ"),),
    ),
    CompanySeed(
        slug="super-micro",
        name="Super Micro Computer",
        sector="Information Technology",
        industry="Computer Hardware",
        country="US",
        website="https://www.supermicro.com",
        description=(
            "Integrates accelerators into rack-scale AI systems. A direct read "
            "on how quickly silicon converts into deployed capacity."
        ),
        tags=(EcosystemTag.SERVERS,),
        tickers=(TickerSeed("SMCI", "Super Micro Computer, Inc.", "NASDAQ"),),
    ),
    CompanySeed(
        slug="ge-vernova",
        name="GE Vernova",
        sector="Industrials",
        industry="Electrical Equipment",
        country="US",
        website="https://www.gevernova.com",
        description=(
            "Supplies turbines and grid equipment for data-centre power. Power "
            "availability, not silicon, is increasingly the limiting factor on "
            "AI capacity growth."
        ),
        tags=(EcosystemTag.POWER_INFRASTRUCTURE,),
        tickers=(TickerSeed("GEV", "GE Vernova Inc.", "NYSE"),),
    ),
)


#: Benchmarks and sector proxies. No parent company: an ETF is a listing only.
#: SMH is the sector benchmark that ``relative_strength_smh`` is computed
#: against; VOO is the broad-market control that separates an AI-specific move
#: from the whole market rising.
ETFS: tuple[TickerSeed, ...] = (
    TickerSeed("SMH", "VanEck Semiconductor ETF", "NASDAQ", asset_type=AssetType.ETF),
    TickerSeed("VOO", "Vanguard S&P 500 ETF", "NYSEARCA", asset_type=AssetType.ETF),
    TickerSeed("SOXX", "iShares Semiconductor ETF", "NASDAQ", asset_type=AssetType.ETF),
)

#: The benchmark other tickers are measured against.
BENCHMARK_SYMBOL = "SMH"
