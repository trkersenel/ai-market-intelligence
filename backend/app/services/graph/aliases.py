"""Alternative names for graph entities, as the press writes them.

The mention detector is only as good as this table. An article saying "Taiwan
Semiconductor Manufacturing said Tuesday" names TSMC, and a detector holding
only the canonical name sees nothing -- which silently removes the article from
extraction rather than failing visibly.

Kept separate from the entity table so adding a name is a one-line edit that
does not touch the seed data's shape.
"""

from __future__ import annotations

#: Surface forms per entity slug. The canonical name and ticker are added
#: automatically and are not repeated here.
ALIASES: dict[str, tuple[str, ...]] = {
    "nvidia": ("Nvidia", "NVIDIA Corp", "NVIDIA Corporation"),
    "amd": ("Advanced Micro Devices", "AMD Inc"),
    "intel": ("Intel Corp", "Intel Corporation"),
    "broadcom": ("Broadcom Inc", "Broadcom Corporation"),
    "marvell": ("Marvell", "Marvell Technology Group"),
    "tsmc": (
        "Taiwan Semiconductor Manufacturing",
        "Taiwan Semiconductor",
        "TSMC",
        "Taiwan Semiconductor Manufacturing Company",
    ),
    "samsung": ("Samsung", "Samsung Electronics Co"),
    "globalfoundries": ("GlobalFoundries", "Global Foundries"),
    "asml": ("ASML Holding", "ASML Holding NV"),
    "applied-materials": ("Applied Materials", "Applied Materials Inc"),
    "lam-research": ("Lam Research", "Lam Research Corp"),
    "kla": ("KLA", "KLA-Tencor", "KLA Corp"),
    "tokyo-electron": ("Tokyo Electron", "TEL"),
    "micron": ("Micron", "Micron Technology Inc"),
    "sk-hynix": ("SK Hynix", "SK hynix", "Hynix", "SKHY"),
    "supermicro": ("Super Micro", "Supermicro", "Super Micro Computer Inc"),
    "dell": ("Dell", "Dell Technologies Inc"),
    "vertiv": ("Vertiv", "Vertiv Holdings Co"),
    "eaton": ("Eaton Corp", "Eaton Corporation"),
    "ge-vernova": ("GE Vernova", "GEV"),
    "arista": ("Arista", "Arista Networks Inc"),
    "coherent": ("Coherent", "Coherent Corp"),
    "microsoft": ("Microsoft", "Microsoft Corp", "Azure", "Microsoft Azure"),
    "amazon": ("Amazon", "Amazon.com", "AWS", "Amazon Web Services"),
    "alphabet": ("Alphabet", "Google", "Google Cloud", "DeepMind"),
    "meta": ("Meta", "Meta Platforms Inc", "Facebook"),
    "oracle": ("Oracle", "Oracle Corp", "OCI"),
    "coreweave": ("CoreWeave", "CoreWeave Inc"),
    "openai": ("OpenAI", "Open AI", "ChatGPT"),
    "anthropic": ("Anthropic", "Claude"),
    "euv-lithography": ("EUV", "extreme ultraviolet"),
    "hbm": ("HBM", "High Bandwidth Memory", "HBM3", "HBM3E", "HBM4"),
    "cowos": ("CoWoS", "chip-on-wafer-on-substrate"),
    "tsmc-arizona": ("TSMC Arizona", "Arizona fab"),
}
