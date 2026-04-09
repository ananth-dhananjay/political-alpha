"""
Political Alpha – Data Processing
Tags disclosures and bills by sector, builds trade and legislative signals.
"""

import pandas as pd
import numpy as np

SECTOR_KEYWORDS = {
    "defense": ["defense", "military", "armed forces", "national security", "veteran",
                "pentagon", "weapons", "navy", "army", "air force", "lockheed",
                "raytheon", "northrop", "boeing", "general dynamics"],
    "finance": ["banking", "financial", "securities", "wall street", "federal reserve",
                "credit", "loan", "mortgage", "insurance", "fintech",
                "jpmorgan", "goldman", "bank of america", "citigroup"],
    "energy":  ["energy", "oil", "gas", "petroleum", "drilling", "pipeline", "solar",
                "wind", "renewable", "nuclear", "coal", "exxon", "chevron", "opec",
                "sanctions", "lng", "fracking"],
    "healthcare": ["health", "medical", "pharma", "drug", "fda", "medicare", "medicaid",
                   "hospital", "vaccine", "biotech", "pfizer", "unitedhealth"],
    "tech":    ["technology", "software", "internet", "cyber", "artificial intelligence",
                "semiconductor", "chip", "data privacy", "antitrust", "apple", "google",
                "microsoft", "amazon", "meta", "nvidia"],
}


def tag_sector(text):
    """Return list of matching sectors for a text string."""
    if not text or not isinstance(text, str):
        return []
    text = text.lower()
    return [s for s, kws in SECTOR_KEYWORDS.items() if any(k in text for k in kws)]
