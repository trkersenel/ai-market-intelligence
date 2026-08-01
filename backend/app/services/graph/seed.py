"""The curated core of the AI infrastructure ecosystem.

Where relationship data comes from is the hardest honest question this platform
faces. PitchBook and Bloomberg SPLC sell exactly this and charge accordingly.
The alternatives are to infer edges from text with a model -- fast, broad, and
wrong often enough to be dangerous -- or to curate a smaller graph from public
disclosure and be explicit about its edges.

This module is the second. Every relationship below is drawn from something a
company said in public: a 10-K supplier concentration disclosure, an earnings
call, an official product page, a press release. The graph it produces is small
and it is *right*, which is the only useful starting point for a system whose
selling point is explaining rather than asserting.

Model-proposed edges are a later layer on top of this one, stored with
``EvidenceSource.INFERRED`` and a low confidence so the two never blur. A
curated backbone is what gives an inference something to be checked against.

**Weights are judgements, and are marked as such.** ``weight`` answers "how much
does this relationship matter to the source", which no filing states directly.
The values here are reasoned from disclosed revenue concentration where that
exists and from public reporting where it does not. They drive impact
propagation, so they are documented rather than tuned until the output looks
nice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.models.enums import EntityKind, EvidenceSource, RelationKind


@dataclass(frozen=True, slots=True)
class SeedEntity:
    """One node in the curated graph."""

    slug: str
    name: str
    kind: EntityKind
    symbol: str | None = None
    country: str | None = None
    tags: tuple[str, ...] = ()
    summary: str | None = None
    #: Other names the press uses. Without these the mention detector finds
    #: "Taiwan Semiconductor Manufacturing" in an article and matches nothing,
    #: because the graph calls it TSMC.
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SeedRelation:
    """One curated edge, with the disclosure it rests on."""

    source: str
    target: str
    kind: RelationKind
    description: str
    weight: float
    confidence: float = 0.95
    source_kind: EvidenceSource = EvidenceSource.CURATED
    citation: str | None = None
    valid_from: date | None = None
    valid_to: date | None = None


C = EntityKind.COMPANY
O = EntityKind.ORGANISATION  # noqa: E741 - reads better than ORG in the table below
T = EntityKind.TECHNOLOGY
F = EntityKind.FACILITY

#: The nodes. Tagged by layer of the stack, which is what makes "every cooling
#: company" or "every foundry" a single indexed query rather than a taxonomy.
ENTITIES: tuple[SeedEntity, ...] = (
    # --- Accelerator designers -------------------------------------------
    SeedEntity(
        "nvidia",
        "NVIDIA",
        C,
        "NVDA",
        "US",
        ("gpu", "networking", "software", "ai-compute"),
        "Designs the accelerators most AI training runs on, and increasingly the "
        "networking and software around them.",
    ),
    SeedEntity(
        "amd",
        "AMD",
        C,
        "AMD",
        "US",
        ("gpu", "cpu", "ai-compute"),
        "The principal merchant alternative to NVIDIA in AI accelerators.",
    ),
    SeedEntity(
        "intel",
        "Intel",
        C,
        "INTC",
        "US",
        ("cpu", "foundry", "ai-compute"),
        "Designs and, unusually among its peers, still fabricates its own chips.",
    ),
    SeedEntity(
        "broadcom",
        "Broadcom",
        C,
        "AVGO",
        "US",
        ("networking", "custom-silicon", "ai-compute"),
        "Builds the custom accelerators hyperscalers use to avoid buying merchant "
        "GPUs, and much of the switching silicon that connects them.",
    ),
    SeedEntity(
        "marvell",
        "Marvell Technology",
        C,
        "MRVL",
        "US",
        ("networking", "custom-silicon"),
        "Custom silicon and optical interconnect for data centres.",
    ),
    # --- Foundry and packaging -------------------------------------------
    SeedEntity(
        "tsmc",
        "TSMC",
        C,
        "TSM",
        "TW",
        ("foundry", "packaging"),
        "Fabricates the leading-edge silicon nearly every AI accelerator is built "
        "on, and owns the advanced packaging capacity that gates HBM assembly.",
    ),
    SeedEntity(
        "samsung",
        "Samsung Electronics",
        C,
        "005930.KS",
        "KR",
        ("foundry", "hbm", "dram"),
        "One of three HBM suppliers and the only credible leading-edge foundry "
        "alternative to TSMC.",
    ),
    SeedEntity(
        "globalfoundries",
        "GlobalFoundries",
        C,
        "GFS",
        "US",
        ("foundry",),
        "Mature-node foundry; not a leading-edge competitor since abandoning 7nm.",
    ),
    # --- Semiconductor equipment -----------------------------------------
    SeedEntity(
        "asml",
        "ASML",
        C,
        "ASML",
        "NL",
        ("semicap", "lithography"),
        "The sole producer of EUV lithography systems, and therefore a single "
        "point of dependency for the entire leading edge.",
    ),
    SeedEntity(
        "applied-materials",
        "Applied Materials",
        C,
        "AMAT",
        "US",
        ("semicap",),
        "Deposition and etch equipment across essentially every fab.",
    ),
    SeedEntity(
        "lam-research",
        "Lam Research",
        C,
        "LRCX",
        "US",
        ("semicap", "etch"),
        "Etch and deposition, with particular exposure to memory capacity.",
    ),
    SeedEntity(
        "kla",
        "KLA Corporation",
        C,
        "KLAC",
        "US",
        ("semicap", "metrology"),
        "Process control and inspection.",
    ),
    SeedEntity(
        "tokyo-electron",
        "Tokyo Electron",
        C,
        "8035.T",
        "JP",
        ("semicap",),
        "Coater/developer and etch equipment; the Japanese counterpart to AMAT.",
    ),
    # --- Memory -----------------------------------------------------------
    SeedEntity(
        "micron",
        "Micron Technology",
        C,
        "MU",
        "US",
        ("dram", "hbm", "nand", "memory"),
        "The only US-headquartered DRAM maker and one of three HBM suppliers.",
    ),
    SeedEntity(
        "sk-hynix",
        "SK Hynix",
        C,
        "000660.KS",
        "KR",
        ("dram", "hbm", "memory"),
        "The HBM share leader through the HBM3 generation.",
    ),
    # --- Systems and infrastructure --------------------------------------
    SeedEntity(
        "supermicro",
        "Super Micro Computer",
        C,
        "SMCI",
        "US",
        ("servers", "systems"),
        "Assembles accelerator servers and racks, including liquid-cooled designs.",
    ),
    SeedEntity(
        "dell",
        "Dell Technologies",
        C,
        "DELL",
        "US",
        ("servers", "systems"),
        "Enterprise AI server integration at scale.",
    ),
    SeedEntity(
        "vertiv",
        "Vertiv Holdings",
        C,
        "VRT",
        "US",
        ("cooling", "power", "datacenter"),
        "Power distribution and thermal management for data centres -- the "
        "constraint that binds once compute is available.",
    ),
    SeedEntity(
        "eaton",
        "Eaton",
        C,
        "ETN",
        "IE",
        ("power", "datacenter"),
        "Electrical distribution equipment for data centre build-outs.",
    ),
    SeedEntity(
        "ge-vernova",
        "GE Vernova",
        C,
        "GEV",
        "US",
        ("power", "generation"),
        "Grid and generation equipment; exposed to data centre load growth.",
    ),
    SeedEntity(
        "arista",
        "Arista Networks",
        C,
        "ANET",
        "US",
        ("networking", "datacenter"),
        "High-speed Ethernet switching for AI clusters.",
    ),
    SeedEntity(
        "coherent",
        "Coherent Corp",
        C,
        "COHR",
        "US",
        ("optics", "networking"),
        "Optical transceivers for cluster interconnect.",
    ),
    # --- Cloud and hyperscalers ------------------------------------------
    SeedEntity(
        "microsoft",
        "Microsoft",
        C,
        "MSFT",
        "US",
        ("cloud", "software", "ai-lab"),
        "Azure is the largest single deployment of AI infrastructure, and the "
        "commercial channel for OpenAI's models.",
    ),
    SeedEntity(
        "amazon",
        "Amazon",
        C,
        "AMZN",
        "US",
        ("cloud", "custom-silicon"),
        "AWS, plus the Trainium and Inferentia accelerators built to reduce its "
        "dependence on merchant GPUs.",
    ),
    SeedEntity(
        "alphabet",
        "Alphabet",
        C,
        "GOOGL",
        "US",
        ("cloud", "custom-silicon", "ai-lab"),
        "Google Cloud, the TPU line, and DeepMind.",
    ),
    SeedEntity(
        "meta",
        "Meta Platforms",
        C,
        "META",
        "US",
        ("cloud", "ai-lab", "custom-silicon"),
        "One of the largest buyers of accelerators, for internal use rather than resale.",
    ),
    SeedEntity(
        "oracle",
        "Oracle",
        C,
        "ORCL",
        "US",
        ("cloud",),
        "OCI has become a significant AI training venue on contracted capacity.",
    ),
    SeedEntity(
        "coreweave",
        "CoreWeave",
        C,
        "CRWV",
        "US",
        ("cloud", "gpu-cloud"),
        "A specialised GPU cloud, structurally dependent on accelerator supply.",
    ),
    # --- AI labs ----------------------------------------------------------
    SeedEntity(
        "openai",
        "OpenAI",
        O,
        None,
        "US",
        ("ai-lab", "foundation-model"),
        "Not publicly listed, and among the most consequential nodes in the graph: "
        "its release cadence moves demand through the entire stack.",
    ),
    SeedEntity(
        "anthropic",
        "Anthropic",
        O,
        None,
        "US",
        ("ai-lab", "foundation-model"),
        "Frontier lab; compute commitments span multiple clouds.",
    ),
    # --- Technologies ------------------------------------------------------
    SeedEntity(
        "euv-lithography",
        "EUV lithography",
        T,
        None,
        None,
        ("lithography", "process"),
        "Extreme ultraviolet patterning. Required below roughly 7nm and available "
        "from exactly one vendor.",
    ),
    SeedEntity(
        "hbm",
        "High Bandwidth Memory",
        T,
        None,
        None,
        ("memory", "process"),
        "Stacked DRAM placed beside the accelerator die. The binding constraint on "
        "accelerator output for much of the current cycle.",
    ),
    SeedEntity(
        "cowos",
        "CoWoS advanced packaging",
        T,
        None,
        None,
        ("packaging", "process"),
        "TSMC's chip-on-wafer-on-substrate packaging, which physically joins logic "
        "and HBM. Capacity here has gated accelerator shipments directly.",
    ),
    # --- Facilities --------------------------------------------------------
    SeedEntity(
        "tsmc-arizona",
        "TSMC Arizona",
        F,
        None,
        "US",
        ("fab", "onshoring"),
        "TSMC's US fabs, the centrepiece of leading-edge capacity outside Taiwan.",
    ),
)

#: The edges. Each carries the disclosure it rests on and a weight explaining
#: how much the relationship matters to the source.
RELATIONS: tuple[SeedRelation, ...] = (
    # --- The critical path: design -> fab -> lithography ------------------
    SeedRelation(
        "tsmc",
        "nvidia",
        RelationKind.MANUFACTURES,
        "Fabricates NVIDIA's data centre GPUs on leading-edge nodes.",
        weight=0.95,
        confidence=0.98,
        citation="NVIDIA 10-K: 'We do not manufacture... we depend on TSMC'.",
    ),
    SeedRelation(
        "tsmc",
        "amd",
        RelationKind.MANUFACTURES,
        "Fabricates AMD's Instinct accelerators and Ryzen/EPYC processors.",
        weight=0.9,
        confidence=0.98,
        citation="AMD 10-K foundry dependence disclosure.",
    ),
    SeedRelation(
        "tsmc",
        "broadcom",
        RelationKind.MANUFACTURES,
        "Fabricates Broadcom's custom accelerators and switching silicon.",
        weight=0.85,
        confidence=0.95,
    ),
    SeedRelation(
        "asml",
        "tsmc",
        RelationKind.SUPPLIES,
        "Sole supplier of EUV lithography systems, without which leading-edge "
        "nodes cannot be patterned at all.",
        weight=0.95,
        confidence=0.99,
        citation="ASML is the only producer of EUV systems worldwide.",
    ),
    SeedRelation(
        "asml",
        "samsung",
        RelationKind.SUPPLIES,
        "EUV and DUV lithography for logic and memory.",
        weight=0.85,
        confidence=0.98,
    ),
    SeedRelation(
        "asml",
        "intel",
        RelationKind.SUPPLIES,
        "EUV systems, including the first High-NA shipments.",
        weight=0.8,
        confidence=0.98,
    ),
    SeedRelation(
        "asml",
        "euv-lithography",
        RelationKind.PRODUCES,
        "The only vendor producing EUV systems.",
        weight=1.0,
        confidence=1.0,
    ),
    SeedRelation(
        "tsmc",
        "euv-lithography",
        RelationKind.USES,
        "Leading-edge nodes are patterned with EUV.",
        weight=0.9,
        confidence=0.99,
    ),
    # --- Equipment into fabs ----------------------------------------------
    SeedRelation(
        "applied-materials",
        "tsmc",
        RelationKind.SUPPLIES,
        "Deposition, etch and ion implantation equipment.",
        weight=0.6,
        confidence=0.9,
    ),
    SeedRelation(
        "lam-research",
        "micron",
        RelationKind.SUPPLIES,
        "Etch and deposition equipment; memory capex is a large share of Lam's demand.",
        weight=0.7,
        confidence=0.9,
    ),
    SeedRelation(
        "lam-research",
        "sk-hynix",
        RelationKind.SUPPLIES,
        "Etch and deposition for DRAM and HBM capacity.",
        weight=0.65,
        confidence=0.9,
    ),
    SeedRelation(
        "kla",
        "tsmc",
        RelationKind.SUPPLIES,
        "Process control and inspection across the fab line.",
        weight=0.55,
        confidence=0.9,
    ),
    SeedRelation(
        "tokyo-electron",
        "tsmc",
        RelationKind.SUPPLIES,
        "Coater/developer tracks paired with lithography.",
        weight=0.6,
        confidence=0.9,
    ),
    # --- Memory into accelerators -----------------------------------------
    SeedRelation(
        "micron",
        "nvidia",
        RelationKind.SUPPLIES,
        "Supplies HBM3E for data centre accelerators.",
        weight=0.8,
        confidence=0.95,
        citation="Micron has publicly confirmed HBM3E qualification and sold-out capacity.",
        valid_from=date(2024, 2, 1),
    ),
    SeedRelation(
        "sk-hynix",
        "nvidia",
        RelationKind.SUPPLIES,
        "The largest HBM supplier to NVIDIA through the HBM3 generation.",
        weight=0.9,
        confidence=0.95,
    ),
    SeedRelation(
        "samsung",
        "nvidia",
        RelationKind.SUPPLIES,
        "HBM supply, qualified later than its two competitors.",
        weight=0.5,
        confidence=0.8,
    ),
    SeedRelation(
        "micron",
        "hbm",
        RelationKind.PRODUCES,
        "Produces HBM3E.",
        weight=0.9,
        confidence=0.99,
    ),
    SeedRelation(
        "sk-hynix",
        "hbm",
        RelationKind.PRODUCES,
        "Produces HBM3 and HBM3E.",
        weight=0.95,
        confidence=0.99,
    ),
    SeedRelation(
        "samsung",
        "hbm",
        RelationKind.PRODUCES,
        "Produces HBM.",
        weight=0.8,
        confidence=0.95,
    ),
    SeedRelation(
        "nvidia",
        "hbm",
        RelationKind.DEPENDS_ON,
        "Every data centre accelerator ships with HBM beside the die; supply has "
        "been the binding constraint on output.",
        weight=0.9,
        confidence=0.98,
    ),
    SeedRelation(
        "nvidia",
        "cowos",
        RelationKind.DEPENDS_ON,
        "CoWoS packaging joins the logic die to its HBM stacks. Packaging capacity "
        "has gated accelerator shipments independently of wafer supply.",
        weight=0.85,
        confidence=0.95,
    ),
    SeedRelation(
        "tsmc",
        "cowos",
        RelationKind.PRODUCES,
        "TSMC owns the great majority of CoWoS capacity.",
        weight=0.9,
        confidence=0.98,
    ),
    # --- Systems -----------------------------------------------------------
    SeedRelation(
        "nvidia",
        "supermicro",
        RelationKind.SUPPLIES,
        "Supplies the accelerators Supermicro integrates into servers and racks.",
        weight=0.85,
        confidence=0.95,
    ),
    SeedRelation(
        "nvidia",
        "dell",
        RelationKind.SUPPLIES,
        "Supplies accelerators for Dell's enterprise AI systems.",
        weight=0.6,
        confidence=0.9,
    ),
    SeedRelation(
        "vertiv",
        "supermicro",
        RelationKind.SUPPLIES,
        "Thermal management and power distribution for high-density racks.",
        weight=0.5,
        confidence=0.8,
    ),
    SeedRelation(
        "arista",
        "microsoft",
        RelationKind.SUPPLIES,
        "Ethernet switching for Azure's data centre fabric.",
        weight=0.6,
        confidence=0.85,
    ),
    SeedRelation(
        "coherent",
        "arista",
        RelationKind.SUPPLIES,
        "Optical transceivers for high-speed interconnect.",
        weight=0.5,
        confidence=0.8,
    ),
    # --- Deployment --------------------------------------------------------
    SeedRelation(
        "microsoft",
        "nvidia",
        RelationKind.CUSTOMER_OF,
        "Among the largest purchasers of NVIDIA accelerators for Azure.",
        weight=0.9,
        confidence=0.95,
    ),
    SeedRelation(
        "amazon",
        "nvidia",
        RelationKind.CUSTOMER_OF,
        "Deploys NVIDIA accelerators on AWS alongside its own silicon.",
        weight=0.8,
        confidence=0.95,
    ),
    SeedRelation(
        "meta",
        "nvidia",
        RelationKind.CUSTOMER_OF,
        "One of the largest buyers of accelerators, for internal workloads.",
        weight=0.85,
        confidence=0.95,
    ),
    SeedRelation(
        "alphabet",
        "nvidia",
        RelationKind.CUSTOMER_OF,
        "Buys accelerators for Google Cloud while deploying TPUs internally.",
        weight=0.7,
        confidence=0.9,
    ),
    SeedRelation(
        "oracle",
        "nvidia",
        RelationKind.CUSTOMER_OF,
        "OCI capacity is built substantially on NVIDIA systems.",
        weight=0.85,
        confidence=0.9,
    ),
    SeedRelation(
        "coreweave",
        "nvidia",
        RelationKind.DEPENDS_ON,
        "A GPU cloud whose entire product is accelerator access; supply and "
        "allocation are existential rather than commercial.",
        weight=0.95,
        confidence=0.95,
    ),
    SeedRelation(
        "broadcom",
        "alphabet",
        RelationKind.PARTNERS_WITH,
        "Co-develops the TPU line.",
        weight=0.8,
        confidence=0.9,
    ),
    SeedRelation(
        "broadcom",
        "meta",
        RelationKind.PARTNERS_WITH,
        "Co-develops custom accelerator silicon.",
        weight=0.6,
        confidence=0.85,
    ),
    # --- Labs and capital --------------------------------------------------
    SeedRelation(
        "microsoft",
        "openai",
        RelationKind.INVESTS_IN,
        "Multi-stage investment alongside a commercial and compute partnership.",
        weight=0.9,
        confidence=0.98,
        valid_from=date(2019, 7, 22),
        citation="Announced publicly by both parties.",
    ),
    SeedRelation(
        "openai",
        "microsoft",
        RelationKind.DEPENDS_ON,
        "Azure has been the primary training and serving venue.",
        weight=0.85,
        confidence=0.9,
        valid_from=date(2019, 7, 22),
    ),
    SeedRelation(
        "amazon",
        "anthropic",
        RelationKind.INVESTS_IN,
        "Investment paired with a compute commitment on AWS.",
        weight=0.7,
        confidence=0.95,
        valid_from=date(2023, 9, 25),
    ),
    SeedRelation(
        "alphabet",
        "anthropic",
        RelationKind.INVESTS_IN,
        "Investment alongside Google Cloud capacity.",
        weight=0.6,
        confidence=0.9,
        valid_from=date(2023, 2, 3),
    ),
    # --- Competition -------------------------------------------------------
    SeedRelation(
        "nvidia",
        "amd",
        RelationKind.COMPETES_WITH,
        "The principal merchant competition in AI accelerators.",
        weight=0.9,
        confidence=0.98,
    ),
    SeedRelation(
        "nvidia",
        "broadcom",
        RelationKind.COMPETES_WITH,
        "Custom hyperscaler silicon displaces merchant GPU demand.",
        weight=0.7,
        confidence=0.85,
    ),
    SeedRelation(
        "nvidia",
        "intel",
        RelationKind.COMPETES_WITH,
        "Gaudi accelerators compete at the margin.",
        weight=0.4,
        confidence=0.85,
    ),
    SeedRelation(
        "tsmc",
        "samsung",
        RelationKind.COMPETES_WITH,
        "The only two credible leading-edge foundries.",
        weight=0.8,
        confidence=0.95,
    ),
    SeedRelation(
        "tsmc",
        "intel",
        RelationKind.COMPETES_WITH,
        "Intel Foundry competes for external leading-edge customers.",
        weight=0.5,
        confidence=0.9,
    ),
    SeedRelation(
        "micron",
        "sk-hynix",
        RelationKind.COMPETES_WITH,
        "DRAM and HBM competition.",
        weight=0.9,
        confidence=0.98,
    ),
    SeedRelation(
        "micron",
        "samsung",
        RelationKind.COMPETES_WITH,
        "DRAM and HBM competition.",
        weight=0.9,
        confidence=0.98,
    ),
    SeedRelation(
        "microsoft",
        "amazon",
        RelationKind.COMPETES_WITH,
        "Cloud competition, increasingly on AI capacity.",
        weight=0.85,
        confidence=0.98,
    ),
    SeedRelation(
        "microsoft",
        "alphabet",
        RelationKind.COMPETES_WITH,
        "Cloud and foundation model competition.",
        weight=0.8,
        confidence=0.95,
    ),
    SeedRelation(
        "openai",
        "anthropic",
        RelationKind.COMPETES_WITH,
        "Frontier model competition.",
        weight=0.85,
        confidence=0.95,
    ),
    # --- Facilities and geography -----------------------------------------
    SeedRelation(
        "tsmc",
        "tsmc-arizona",
        RelationKind.OPERATES,
        "TSMC's US leading-edge capacity.",
        weight=0.3,
        confidence=0.99,
        valid_from=date(2024, 4, 1),
    ),
    SeedRelation(
        "amd",
        "tsmc",
        RelationKind.DEPENDS_ON,
        "Fabless: has no leading-edge capacity of its own.",
        weight=0.9,
        confidence=0.98,
    ),
    SeedRelation(
        "nvidia",
        "tsmc",
        RelationKind.DEPENDS_ON,
        "Fabless: leading-edge supply is concentrated in a single foundry in a "
        "single jurisdiction.",
        weight=0.95,
        confidence=0.98,
    ),
)
