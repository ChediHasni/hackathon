"""
ZoneScore — Criteria configuration with multilingual synonyms.

Each criterion maps French, Arabic and English phrases to a criterion id, so the
LLM can understand expressions like "calme", "هادئ", "quiet" → noise/silence,
or "propre", "نظيف", "clean" → air quality.
"""

from dataclasses import dataclass
from typing import Dict, List

# Importance level → numeric weight.  The LLM expresses importance as levels;
# the code maps them to numbers.  This is far more stable than raw floats.
IMPORTANCE_WEIGHTS = {
    "low": 0.25,
    "medium": 0.5,
    "high": 1.0,
    "must_have": 2.0,
}

# Default weight for criteria the user never mentioned (applied to avg_score).
DEFAULT_WEIGHT = 0.1

# Hard minimum threshold used when a criterion is marked "must_have" but no
# explicit min value is given: the zone must score at least this (0-100).
MUST_HAVE_DEFAULT_MIN = 60.0


@dataclass(frozen=True)
class Criterion:
    id: str
    name: str                       # short display name
    description: str                # what it measures
    high_meaning: str               # what a high score means
    low_meaning: str                # what a low score means
    synonyms_fr: List[str]          # French cue phrases
    synonyms_ar: List[str]          # Arabic cue phrases (Arabic script)
    synonyms_en: List[str]          # English cue phrases
    default_min: float = 0.0        # hard minimum if user sets a floor (0-100)


# ─── Criteria list ──────────────────────────────────────────────────────────
# The ids must match the criterion keys produced by the GNN pipeline so that
# per-zone scores can be looked up directly.
CRITERIA_CONFIG: Dict[str, Criterion] = {
    "greenness": Criterion(
        id="greenness",
        name="Greenness",
        description="Vegetation coverage (trees, parks, gardens) from satellite NDVI.",
        high_meaning="Very green, many trees and parks",
        low_meaning="Concrete-heavy, little vegetation",
        synonyms_fr=[
            "vert", "verte", "verdure", "arbres", "parc", "jardin", "nature",
            "espaces verts", "boisé", "boisée", "feuillu", "végétation",
        ],
        synonyms_ar=[
            "أخضر", "خضراء", "أشجار", "حديقة", "حدائق", "طبيعة", "مساحات خضراء",
            "مغرس", "غابي", "نباتات", "خضرة", "ورق", "شجر",
        ],
        synonyms_en=[
            "green", "greenery", "trees", "park", "garden", "nature", "leafy",
            "vegetation", "green space", "wooded", "verdant",
        ],
        default_min=0.0,
    ),
    "climate": Criterion(
        id="climate",
        name="Climate Comfort",
        description="Temperature and precipitation comfort (cooler, moderate rain).",
        high_meaning="Mild temperatures, pleasant weather",
        low_meaning="Very hot or very dry / extreme climate",
        synonyms_fr=[
            "climat", "tempéré", "doux", "frais", "agréable", "météo", "température",
            "pluie", "ventilé", "respirable", "climatisation", "vent", "brise",
        ],
        synonyms_ar=[
            "مناخ", "معتدل", "بارد", "لطيف", "طقس", "درجة حرارة", "مطر", "هواء",
            "نسيم", "مريح", "منعش", "تدفئة", "تكييف", "نسيم",
        ],
        synonyms_en=[
            "climate", "temperate", "mild", "cool", "pleasant", "weather", "temperature",
            "rain", "breezy", "comfortable", "air conditioning", "wind", "breeze",
        ],
        default_min=0.0,
    ),
    "building_svf": Criterion(
        id="building_svf",
        name="Sky View Factor",
        description="Open sky availability — low building density / tall buildings block less sky.",
        high_meaning="Open sky, not overshadowed by tall buildings",
        low_meaning="Canyon-like streets, dense tall buildings blocking sky",
        synonyms_fr=[
            "ciel", "ouvert", "espace", "dégagé", "pas d'immeubles", "basse", "faible",
            "panoramique", "horizon", "lumière", "ensoleillé", "large", "espacieux",
        ],
        synonyms_ar=[
            "سماء", "مفتوح", "مساحة", "مشمس", "بدون أبنية", "منخفض", "قليل",
            "بانورامي", "أفق", "ضوء", "مشمس", "واسع", "فسيح",
        ],
        synonyms_en=[
            "sky", "open", "spacious", "unobstructed", "no tall buildings", "low-rise",
            "panoramic", "horizon", "sunlight", "sunny", "wide", "roomy",
        ],
        default_min=0.0,
    ),
    "air_quality": Criterion(
        id="air_quality",
        name="Air Quality",
        description="Clean air — low PM2.5 / NO2 pollution.",
        high_meaning="Clean, fresh air, little pollution",
        low_meaning="Polluted, smoggy, near traffic or industry",
        synonyms_fr=[
            "air", "propre", "pur", "sain", "pollution", "pas pollué", "frais",
            "respirable", "sans fumée", "écologique", "qualité de l'air",
        ],
        synonyms_ar=[
            "هواء", "نظيف", "نقي", "صحي", "تلوث", "غير ملوث", "عذب", "تنفس",
            "بدون دخان", "بيئي", "جودة الهواء", "عطر", "نعيم",
        ],
        synonyms_en=[
            "air", "clean", "pure", "healthy", "pollution", "unpolluted", "fresh",
            "breathable", "no smoke", "eco-friendly", "air quality",
        ],
        default_min=0.0,
    ),
    "heat": Criterion(
        id="heat",
        name="Heat / LST",
        description="Land surface temperature — cooler zones score higher.",
        high_meaning="Cool, shaded, not overheated",
        low_meaning="Hot, urban heat island, sun-exposed",
        synonyms_fr=[
            "frais", "froide", "chaleur", "canicule", "ombre", "refroidi", "tempéré",
            "pas chaud", "ventilé", "îlot de chaleur", "température", "climatiseur",
        ],
        synonyms_ar=[
            "بارد", "برودة", "حر", "موجة حر", "ظل", "معتدل", "غير حار", "متنسم",
            "جزيرة حرارية", "درجة حرارة", "تكييف", "تدفئة", "نسيم",
        ],
        synonyms_en=[
            "cool", "cold", "heat", "heatwave", "shade", "cooled", "temperate",
            "not hot", "ventilated", "heat island", "temperature", "air conditioning",
        ],
        default_min=0.0,
    ),
    "accessibility": Criterion(
        id="accessibility",
        name="Accessibility",
        description="Proximity to schools, hospitals, pharmacies, supermarkets (OSM POIs).",
        high_meaning="Many services nearby, walkable",
        low_meaning="Few services, car-dependent, isolated",
        synonyms_fr=[
            "commerces", "services", "proche", "à pied", "école", "hôpital", "pharmacie",
            "supermarché", "accessible", "commodités", "proximité", "magasins", "centre",
        ],
        synonyms_ar=[
            "محلات", "خدمات", "قريب", "مشي", "مدرسة", "مستشفى", "صيدلية", "سوبرماركت",
            "متاح", "مرافق", "قرب", "متاجر", "مركز", "قريب منك",
        ],
        synonyms_en=[
            "shops", "services", "close", "walking", "school", "hospital", "pharmacy",
            "supermarket", "accessible", "amenities", "proximity", "stores", "center",
        ],
        default_min=0.0,
    ),
    "transit": Criterion(
        id="transit",
        name="Transit Access",
        description="Public transport stops per zone (light metro, bus, train).",
        high_meaning="Well served by public transport, many stops",
        low_meaning="Poorly served, few or no stops",
        synonyms_fr=[
            "transport", "métro", "bus", "tram", "gare", "arrêt", "station", "transit",
            "déplacement", "navette", "TCSP", "métro léger", "lignes", "réseau",
        ],
        synonyms_ar=[
            "نقل", "مترو", "حافلة", "ترام", "محطة", "موقف", "مترو أنفاق", "مواصلات",
            "انتقال", "حافلة صغيرة", "خطوط", "شبكة", "قطار", "تسوق",
        ],
        synonyms_en=[
            "transport", "metro", "bus", "tram", "train", "station", "stop", "transit",
            "commute", "shuttle", "lines", "network", "railway", "public transport",
        ],
        default_min=0.0,
    ),
}

CRITERION_IDS = list(CRITERIA_CONFIG.keys())


def get_criterion_synonyms(criterion_id: str) -> Dict[str, List[str]]:
    """Return {fr, ar, en} synonym lists for a criterion id."""
    c = CRITERIA_CONFIG[criterion_id]
    return {"fr": c.synonyms_fr, "ar": c.synonyms_ar, "en": c.synonyms_en}


def all_synonyms_flat() -> List[str]:
    """All synonym phrases across all criteria and languages (for keyword hints)."""
    phrases: List[str] = []
    for c in CRITERIA_CONFIG.values():
        phrases.extend(c.synonyms_fr)
        phrases.extend(c.synonyms_ar)
        phrases.extend(c.synonyms_en)
    return phrases