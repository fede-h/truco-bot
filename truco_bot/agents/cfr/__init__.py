from truco_bot.agents.cfr.fallback import EquityFallback
from truco_bot.agents.cfr.history import HandHistoryTracker
from truco_bot.agents.cfr.isomorphism import (
    canonical_deal_partitions,
    canonical_infoset_key,
    canonicalize_hand,
    generate_canonical_deals,
)
from truco_bot.agents.cfr.native_agent import NativeCFRAgent
from truco_bot.agents.cfr.train import train_and_save

# Primary CFR agent alias
CFRAgent = NativeCFRAgent

__all__ = [
    "CFRAgent",
    "EquityFallback",
    "HandHistoryTracker",
    "NativeCFRAgent",
    "canonical_deal_partitions",
    "canonical_infoset_key",
    "canonicalize_hand",
    "generate_canonical_deals",
    "train_and_save",
]


