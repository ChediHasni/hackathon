"""
ZoneScore — Tunisia zone / neighbourhood registry.

Defines Tunisian administrative neighbourhoods with real coordinates and H3 cells.
Per-criterion scores and overall ZoneGAT quality scores are populated dynamically
from the live preprocessing + GNN pipeline via `api.live_zones` (never hardcoded).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import h3 as h3lib


@dataclass
class Zone:
    id: str
    name: str
    name_ar: str
    governorate: str
    governorate_ar: str
    delegation: str
    delegation_ar: str
    lat: float
    lon: float
    population_estimate: int
    h3_cells: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    avg_score: float = 0.0
    data_sources: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.h3_cells:
            center_cell = h3lib.latlng_to_cell(self.lat, self.lon, 7)
            self.h3_cells = [center_cell]

    def to_dict(self, include_scores: bool = True) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "name_ar": self.name_ar,
            "governorate": self.governorate,
            "governorate_ar": self.governorate_ar,
            "delegation": self.delegation,
            "delegation_ar": self.delegation_ar,
            "lat": self.lat,
            "lon": self.lon,
            "avg_score": round(self.avg_score, 1),
            "population_estimate": self.population_estimate,
            "h3_cells": self.h3_cells,
        }
        if include_scores:
            d["scores"] = {k: round(v, 1) for k, v in self.scores.items()}
            if self.data_sources:
                d["data_sources"] = dict(self.data_sources)
        return d

    def strongest_criterion(self) -> Optional[str]:
        if not self.scores:
            return None
        return max(self.scores, key=self.scores.get)

    def weakest_criterion(self) -> Optional[str]:
        if not self.scores:
            return None
        return min(self.scores, key=self.scores.get)


# ─── Tunisia neighbourhoods (real coordinates; scores populated from live APIs + GNN) ─
ZONES: List[Zone] = [
    Zone(
        id="zone_tunis_medina",
        name="Medina de Tunis",
        name_ar="المدينة العتيقة بتونس",
        governorate="Tunis",
        governorate_ar="تونس",
        delegation="Tunis Centre",
        delegation_ar="تونس المدينة",
        lat=36.7975, lon=10.1712,
        population_estimate=85000,
    ),
    Zone(
        id="zone_tunis_bardo",
        name="Le Bardo",
        name_ar="الباردو",
        governorate="Tunis",
        governorate_ar="تونس",
        delegation="Le Bardo",
        delegation_ar="الباردو",
        lat=36.8087, lon=10.1386,
        population_estimate=75000,
    ),
    Zone(
        id="zone_lamarsa",
        name="La Marsa",
        name_ar="المرسى",
        governorate="Tunis",
        governorate_ar="تونس",
        delegation="La Marsa",
        delegation_ar="المرسى",
        lat=36.8814, lon=10.3231,
        population_estimate=95000,
    ),
    Zone(
        id="zone_carthage",
        name="Carthage",
        name_ar="قرطاج",
        governorate="Tunis",
        governorate_ar="تونس",
        delegation="Carthage",
        delegation_ar="قرطاج",
        lat=36.8521, lon=10.3256,
        population_estimate=22000,
    ),
    Zone(
        id="zone_sidibousaid",
        name="Sidi Bou Said",
        name_ar="سيدي بوسعيد",
        governorate="Tunis",
        governorate_ar="تونس",
        delegation="Sidi Bou Said",
        delegation_ar="سيدي بوسعيد",
        lat=36.8690, lon=10.3470,
        population_estimate=6000,
    ),
    Zone(
        id="zone_ariana_centre",
        name="Ariana Centre",
        name_ar="أريانة المركز",
        governorate="Ariana",
        governorate_ar="أريانة",
        delegation="Ariana Ville",
        delegation_ar="أريانة المدينة",
        lat=36.8618, lon=10.1926,
        population_estimate=110000,
    ),
    Zone(
        id="zone_manouba",
        name="Manouba",
        name_ar="منوبة",
        governorate="Manouba",
        governorate_ar="منوبة",
        delegation="Manouba",
        delegation_ar="منوبة",
        lat=36.8089, lon=10.1008,
        population_estimate=55000,
    ),
    Zone(
        id="zone_benarous",
        name="Ben Arous",
        name_ar="بن عروس",
        governorate="Ben Arous",
        governorate_ar="بن عروس",
        delegation="Ben Arous",
        delegation_ar="بن عروس",
        lat=36.7570, lon=10.2276,
        population_estimate=105000,
    ),
    Zone(
        id="zone_nabeul",
        name="Nabeul",
        name_ar="نابل",
        governorate="Nabeul",
        governorate_ar="نابل",
        delegation="Nabeul",
        delegation_ar="نابل",
        lat=36.4567, lon=10.7378,
        population_estimate=75000,
    ),
    Zone(
        id="zone_sousse_centre",
        name="Sousse Centre",
        name_ar="سوسة المركز",
        governorate="Sousse",
        governorate_ar="سوسة",
        delegation="Sousse Ville",
        delegation_ar="سوسة المدينة",
        lat=35.8256, lon=10.6366,
        population_estimate=220000,
    ),
    Zone(
        id="zone_sfax_centre",
        name="Sfax Centre",
        name_ar="صفاقس المركز",
        governorate="Sfax",
        governorate_ar="صفاقس",
        delegation="Sfax Ville",
        delegation_ar="صفاقس المدينة",
        lat=34.7406, lon=10.7604,
        population_estimate=330000,
    ),
    Zone(
        id="zone_hammanetz",
        name="Hammamet",
        name_ar="حمامات",
        governorate="Nabeul",
        governorate_ar="نابل",
        delegation="Hammamet",
        delegation_ar="حمامات",
        lat=36.3983, lon=10.6153,
        population_estimate=70000,
    ),
    Zone(
        id="zone_ouedlil",
        name="Oued Lil",
        name_ar="وادي اللّيل",
        governorate="Manouba",
        governorate_ar="منوبة",
        delegation="Oued Ellil",
        delegation_ar="وادي اللّيل",
        lat=36.8328, lon=10.0475,
        population_estimate=30000,
    ),
]

# ─── Lookup indexes ─────────────────────────────────────────────────────────
ZONE_BY_ID: Dict[str, Zone] = {z.id: z for z in ZONES}

ZONES_BY_GOVERNORATE: Dict[str, List[Zone]] = {}
ZONES_BY_DELEGATION: Dict[str, List[Zone]] = {}
for z in ZONES:
    ZONES_BY_GOVERNORATE.setdefault(z.governorate, []).append(z)
    ZONES_BY_DELEGATION.setdefault(z.delegation, []).append(z)

GOVERNORATES = sorted(ZONES_BY_GOVERNORATE.keys())
DELEGATIONS = sorted(ZONES_BY_DELEGATION.keys())

# All searchable name variants for the location resolver.
_LOCATION_ALIASES: List[tuple[str, str]] = []  # (display_text, zone_id)
for z in ZONES:
    _LOCATION_ALIASES.append((z.name, z.id))
    _LOCATION_ALIASES.append((z.name_ar, z.id))
    _LOCATION_ALIASES.append((z.governorate, z.id))
    _LOCATION_ALIASES.append((z.governorate_ar, z.id))
    _LOCATION_ALIASES.append((z.delegation, z.id))
    _LOCATION_ALIASES.append((z.delegation_ar, z.id))
