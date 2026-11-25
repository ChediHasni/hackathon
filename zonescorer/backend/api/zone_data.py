"""
ZoneScore — Tunisia zone / neighbourhood reference data.

Each zone represents a neighbourhood (or aggregation of H3 hexagons) with:
  - id, name (FR), governorate, delegation
  - per-criterion scores (0-100) averaged from underlying H3 hexagons
  - avg_score: overall GNN quality score
  - lat, lon: centroid for map display
  - population_estimate: rough population

In production these would come from a PostGIS/GeoDjango table of administrative
boundaries joined to the GNN hexagon scores.  Here we use synthetic but
realistic data so the chat system is testable immediately.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from .criteria_config import CRITERION_IDS, DEFAULT_WEIGHT


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
    scores: Dict[str, float]      # criterion_id -> score (0-100)
    avg_score: float              # overall GNN score (0-100)
    population_estimate: int
    h3_cells: List[str]           # underlying H3 hexagon ids (sample)

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
        return d

    def strongest_criterion(self) -> Optional[str]:
        if not self.scores:
            return None
        return max(self.scores, key=self.scores.get)

    def weakest_criterion(self) -> Optional[str]:
        if not self.scores:
            return None
        return min(self.scores, key=self.scores.get)


# ─── Tunisia neighbourhoods ─────────────────────────────────────────────────
ZONES: List[Zone] = [
    Zone(
        id="zone_tunis_medina",
        name="Medina de Tunis",
        name_ar="المدينة العتيقة بتونس",
        governorate="Tunis",
        governorate_ar="تونس",
        delegation="Tunis Centre",
        delegation_ar="تونس中心",
        lat=36.7975, lon=10.1712,
        scores={
            "greenness": 25.0, "climate": 70.0, "building_svf": 30.0,
            "air_quality": 55.0, "heat": 65.0, "accessibility": 95.0, "transit": 90.0,
        },
        avg_score=65.0,
        population_estimate=85000,
        h3_cells=["87283082bffffff", "872830833ffffff"],
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
        scores={
            "greenness": 60.0, "climate": 75.0, "building_svf": 55.0,
            "air_quality": 70.0, "heat": 70.0, "accessibility": 80.0, "transit": 75.0,
        },
        avg_score=72.0,
        population_estimate=75000,
        h3_cells=["87283082a7fffff", "87283082affffff"],
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
        scores={
            "greenness": 85.0, "climate": 85.0, "building_svf": 70.0,
            "air_quality": 80.0, "heat": 75.0, "accessibility": 70.0, "transit": 60.0,
        },
        avg_score=84.0,
        population_estimate=95000,
        h3_cells=["87283087b3fffff", "87283087affffff"],
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
        scores={
            "greenness": 90.0, "climate": 88.0, "building_svf": 85.0,
            "air_quality": 85.0, "heat": 80.0, "accessibility": 50.0, "transit": 45.0,
        },
        avg_score=87.0,
        population_estimate=22000,
        h3_cells=["87283087b1fffff", "87283087b2fffff"],
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
        scores={
            "greenness": 80.0, "climate": 85.0, "building_svf": 75.0,
            "air_quality": 82.0, "heat": 78.0, "accessibility": 55.0, "transit": 40.0,
        },
        avg_score=82.0,
        population_estimate=6000,
        h3_cells=["87283087b4fffff", "87283087b5fffff"],
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
        scores={
            "greenness": 55.0, "climate": 75.0, "building_svf": 45.0,
            "air_quality": 65.0, "heat": 68.0, "accessibility": 85.0, "transit": 70.0,
        },
        avg_score=70.0,
        population_estimate=110000,
        h3_cells=["87283082a3fffff", "87283082a5fffff"],
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
        scores={
            "greenness": 65.0, "climate": 78.0, "building_svf": 60.0,
            "air_quality": 72.0, "heat": 72.0, "accessibility": 70.0, "transit": 55.0,
        },
        avg_score=73.0,
        population_estimate=55000,
        h3_cells=["8728308293fffff", "8728308295fffff"],
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
        scores={
            "greenness": 40.0, "climate": 72.0, "building_svf": 45.0,
            "air_quality": 50.0, "heat": 55.0, "accessibility": 82.0, "transit": 80.0,
        },
        avg_score=64.0,
        population_estimate=105000,
        h3_cells=["87283081b3fffff", "87283081b5fffff"],
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
        scores={
            "greenness": 70.0, "climate": 80.0, "building_svf": 65.0,
            "air_quality": 75.0, "heat": 68.0, "accessibility": 75.0, "transit": 50.0,
        },
        avg_score=75.0,
        population_estimate=75000,
        h3_cells=["8728308b93fffff", "8728308b95fffff"],
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
        scores={
            "greenness": 45.0, "climate": 82.0, "building_svf": 50.0,
            "air_quality": 60.0, "heat": 60.0, "accessibility": 88.0, "transit": 85.0,
        },
        avg_score=70.0,
        population_estimate=220000,
        h3_cells=["87283079a3fffff", "87283079a5fffff"],
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
        scores={
            "greenness": 35.0, "climate": 80.0, "building_svf": 40.0,
            "air_quality": 48.0, "heat": 50.0, "accessibility": 90.0, "transit": 82.0,
        },
        avg_score=63.0,
        population_estimate=330000,
        h3_cells=["87283075a3fffff", "87283075a5fffff"],
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
        scores={
            "greenness": 75.0, "climate": 85.0, "building_svf": 70.0,
            "air_quality": 78.0, "heat": 72.0, "accessibility": 65.0, "transit": 55.0,
        },
        avg_score=79.0,
        population_estimate=70000,
        h3_cells=["8728308b97fffff", "8728308b99fffff"],
    ),
    Zone(
        id="zone_ouedlil",
        name="Oued Lil",
        name_ar="وادي اللّيل",
        governorate="Ben Arous",
        governorate_ar="بن عروس",
        delegation="Oued Lil",
        delegation_ar="وادي اللّيل",
        lat=36.7200, lon=10.2500,
        scores={
            "greenness": 70.0, "climate": 78.0, "building_svf": 75.0,
            "air_quality": 78.0, "heat": 75.0, "accessibility": 50.0, "transit": 40.0,
        },
        avg_score=76.0,
        population_estimate=30000,
        h3_cells=["87283081b7fffff", "87283081b9fffff"],
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