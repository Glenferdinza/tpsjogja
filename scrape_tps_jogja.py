from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests

# Bounding boxes: DIY Yogyakarta and major subregions (approx)
BBOXES = [
    {
        "name": "DIY Yogyakarta",
        "district": "DIY",
        "min_lat": -8.30,
        "max_lat": -7.50,
        "min_lon": 110.00,
        "max_lon": 110.85,
    },
    {
        "name": "Kab. Sleman",
        "district": "Sleman",
        "min_lat": -7.86,
        "max_lat": -7.53,
        "min_lon": 110.25,
        "max_lon": 110.53,
    },
    {
        "name": "Kota Yogyakarta",
        "district": "Kota Yogyakarta",
        "min_lat": -7.86,
        "max_lat": -7.75,
        "min_lon": 110.33,
        "max_lon": 110.43,
    },
    {
        "name": "Kab. Bantul",
        "district": "Bantul",
        "min_lat": -8.14,
        "max_lat": -7.79,
        "min_lon": 110.20,
        "max_lon": 110.51,
    },
    {
        "name": "Kab. Kulon Progo",
        "district": "Kulon Progo",
        "min_lat": -8.22,
        "max_lat": -7.60,
        "min_lon": 109.96,
        "max_lon": 110.30,
    },
    {
        "name": "Kab. Gunungkidul",
        "district": "Gunungkidul",
        "min_lat": -8.26,
        "max_lat": -7.68,
        "min_lon": 110.37,
        "max_lon": 111.00,
    },
]

BBOX_NGAGLIK = {
    "name": "Ngaglik, Sleman",
    "min_lat": -7.76,
    "max_lat": -7.68,
    "min_lon": 110.34,
    "max_lon": 110.44,
}

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
GOOGLE_TEXTSEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"

FACILITY_KEYWORDS = [
    "TPS",
    "TPA",
    "TPST",
    "bank sampah",
    "tempat pembuangan",
    "tempat pembuangan akhir",
    "3R",
]

PRIORITY_SITE_QUERIES = [
    "TPS 3R Brama Muda Ngaglik Sleman",
    "TPS Sumberan Sariharjo Ngaglik Sleman",
    "TPS Al Azhar Ngaglik Sleman",
    "TPS Pondok Ngaglik Sleman",
    "TPS Kasturi Ngaglik Sleman",
    "TPS 3R Ngaglik Sleman",
]

NOISE_KEYWORDS = [
    "satpam",
    "hotel",
    "rumah berhantu",
    "sekolah",
    "masjid",
    "taman pendidikan",
    "day care",
    "pos satpam",
]

BLOCKED_AMENITIES = {
    "school",
    "kindergarten",
    "place_of_worship",
    "community_centre",
    "police",
    "ranger_station",
}


@dataclass
class Point:
    name: str
    lat: float
    lon: float
    amenity: str
    object_type: str
    object_id: str
    source: str


def _bbox_to_viewbox(bbox: Dict[str, float]) -> str:
    return f"{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']},{bbox['min_lat']}"


def overpass_query_for_bbox(bbox: Dict[str, float]) -> str:
    return f"""
[out:json][timeout:35];
(
  node["amenity"~"waste_disposal|recycling|waste_transfer_station"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
  way["amenity"~"waste_disposal|recycling|waste_transfer_station"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
  node["name"~"TPS|TPA|TPST|Bank Sampah|Tempat Pembuangan|3R",i]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
  way["name"~"TPS|TPA|TPST|Bank Sampah|Tempat Pembuangan|3R",i]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
);
out center tags;
""".strip()


def fetch_overpass_points(bbox: Dict[str, float], source_name: str) -> List[Point]:
    query = overpass_query_for_bbox(bbox)
    headers = {"User-Agent": "tps-ngaglik-scraper/1.0"}
    last_error = None
    out: List[Point] = []

    for url in OVERPASS_URLS:
        try:
            response = requests.post(url, data={"data": query}, headers=headers, timeout=60)
            response.raise_for_status()
            payload = response.json()

            for el in payload.get("elements", []):
                tags = el.get("tags", {})
                lat = el.get("lat")
                lon = el.get("lon")
                if lat is None or lon is None:
                    center = el.get("center", {})
                    lat = center.get("lat")
                    lon = center.get("lon")
                if lat is None or lon is None:
                    continue

                name = tags.get("name") or tags.get("operator") or "Unnamed TPS"
                amenity = tags.get("amenity", "unknown")
                out.append(
                    Point(
                        name=str(name).strip(),
                        lat=float(lat),
                        lon=float(lon),
                        amenity=str(amenity),
                        object_type=str(el.get("type", "unknown")),
                        object_id=str(el.get("id", "")),
                        source=f"overpass:{source_name}",
                    )
                )
            if out:
                return out
        except Exception as exc:
            last_error = exc

    if last_error:
        raise RuntimeError(f"Overpass gagal untuk {source_name}: {last_error}")
    return out


def fetch_nominatim_points(bbox: Dict[str, float], source_name: str) -> List[Point]:
    headers = {"User-Agent": "tps-ngaglik-scraper/1.0"}
    viewbox = _bbox_to_viewbox(bbox)
    queries = [
        f"TPS {source_name}",
        f"TPA {source_name}",
        f"TPST {source_name}",
        f"bank sampah {source_name}",
        f"tempat pembuangan sampah {source_name}",
    ]

    out: List[Point] = []
    for q in queries:
        params = {
            "q": q,
            "format": "jsonv2",
            "limit": 40,
            "bounded": 1,
            "viewbox": viewbox,
            "addressdetails": 0,
        }
        try:
            response = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            items = response.json()
            for item in items:
                display_name = str(item.get("display_name", "Unnamed TPS"))
                name = display_name.split(",")[0].strip() or "Unnamed TPS"
                out.append(
                    Point(
                        name=name,
                        lat=float(item["lat"]),
                        lon=float(item["lon"]),
                        amenity=str(item.get("type", "unknown")),
                        object_type="nominatim",
                        object_id=str(item.get("osm_id", "")),
                        source=f"nominatim:{source_name}",
                    )
                )
        except Exception:
            continue
    return out


def fetch_nominatim_priority_sites() -> List[Point]:
    headers = {"User-Agent": "tps-ngaglik-scraper/1.0"}
    viewbox = _bbox_to_viewbox(BBOXES[0])
    out: List[Point] = []

    for q in PRIORITY_SITE_QUERIES:
        params = {
            "q": q,
            "format": "jsonv2",
            "limit": 5,
            "bounded": 1,
            "viewbox": viewbox,
            "addressdetails": 0,
        }
        try:
            response = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            for item in response.json():
                display_name = str(item.get("display_name", "Unnamed TPS"))
                name = display_name.split(",")[0].strip() or "Unnamed TPS"
                out.append(
                    Point(
                        name=name,
                        lat=float(item["lat"]),
                        lon=float(item["lon"]),
                        amenity=str(item.get("type", "unknown")),
                        object_type="nominatim",
                        object_id=str(item.get("osm_id", "")),
                        source="nominatim:priority_sites",
                    )
                )
        except Exception:
            continue

    return out


def fetch_google_places_points() -> List[Point]:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        return []

    queries = [
        "TPS di Yogyakarta",
        "TPA di Yogyakarta",
        "TPST di Yogyakarta",
        "bank sampah Sleman",
        "TPS Ngaglik Sleman",
    ]
    out: List[Point] = []

    for q in queries:
        params = {
            "query": q,
            "region": "id",
            "language": "id",
            "key": api_key,
        }
        page_count = 0
        while True:
            response = requests.get(GOOGLE_TEXTSEARCH_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()

            for r in payload.get("results", []):
                loc = r.get("geometry", {}).get("location", {})
                lat = loc.get("lat")
                lon = loc.get("lng")
                if lat is None or lon is None:
                    continue
                out.append(
                    Point(
                        name=str(r.get("name", "Unnamed TPS")),
                        lat=float(lat),
                        lon=float(lon),
                        amenity="google_place",
                        object_type="google_place",
                        object_id=str(r.get("place_id", "")),
                        source="google_places:textsearch",
                    )
                )

            token = payload.get("next_page_token")
            page_count += 1
            if not token or page_count >= 2:
                break
            time.sleep(2.2)
            params = {"pagetoken": token, "key": api_key}

    return out


def clean_and_enrich(points: Iterable[Point]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "name": p.name,
                "lat": p.lat,
                "lon": p.lon,
                "amenity": p.amenity,
                "object_type": p.object_type,
                "object_id": p.object_id,
                "source": p.source,
            }
            for p in points
        ]
    )

    if df.empty:
        return df

    df["name"] = df["name"].fillna("Unnamed TPS").astype(str).str.strip()
    df = df.dropna(subset=["lat", "lon"]).copy()
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)].copy()

    keyword_re = re.compile(r"(TPS|TPA|TPST|Bank\s+Sampah|Tempat\s+Pembuangan|3R)", re.IGNORECASE)
    noise_re = re.compile("|".join(NOISE_KEYWORDS), re.IGNORECASE)

    def keep_row(row: pd.Series) -> bool:
        name = str(row["name"])
        amenity = str(row["amenity"])
        if noise_re.search(name):
            return False
        if amenity in BLOCKED_AMENITIES:
            return False
        if amenity in {"waste_disposal", "recycling", "waste_transfer_station", "google_place"}:
            return True
        return bool(keyword_re.search(name))

    df = df[df.apply(keep_row, axis=1)].copy()

    df["is_ngaglik_bbox"] = (
        df["lat"].between(BBOX_NGAGLIK["min_lat"], BBOX_NGAGLIK["max_lat"])
        & df["lon"].between(BBOX_NGAGLIK["min_lon"], BBOX_NGAGLIK["max_lon"])
    )

    df["name_key"] = df["name"].str.lower().str.replace(r"\s+", " ", regex=True)
    df["lat_key"] = df["lat"].round(5)
    df["lon_key"] = df["lon"].round(5)
    df = df.sort_values(by=["is_ngaglik_bbox", "source"], ascending=[False, True])
    df = df.drop_duplicates(subset=["name_key", "lat_key", "lon_key"], keep="first")

    def infer_district(lat: float, lon: float) -> str:
        if BBOX_NGAGLIK["min_lat"] <= lat <= BBOX_NGAGLIK["max_lat"] and BBOX_NGAGLIK["min_lon"] <= lon <= BBOX_NGAGLIK["max_lon"]:
            return "Ngaglik"
        for box in BBOXES:
            if box["district"] == "DIY":
                continue
            if box["min_lat"] <= lat <= box["max_lat"] and box["min_lon"] <= lon <= box["max_lon"]:
                return str(box["district"])
        return "DIY"

    df["city"] = "DIY Yogyakarta"
    df["district_estimate"] = [infer_district(float(r["lat"]), float(r["lon"])) for _, r in df.iterrows()]
    df["priority_hint"] = df["amenity"].map(
        {
            "waste_transfer_station": "high",
            "waste_disposal": "high",
            "recycling": "medium",
            "google_place": "medium",
        }
    ).fillna("medium")

    return df[
        [
            "name",
            "lat",
            "lon",
            "amenity",
            "source",
            "city",
            "district_estimate",
            "priority_hint",
            "is_ngaglik_bbox",
            "object_type",
            "object_id",
        ]
    ].reset_index(drop=True)


def run_scrape(output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_points: List[Point] = []
    errors: List[str] = []

    for bbox in [BBOX_NGAGLIK] + BBOXES:
        try:
            raw_points.extend(fetch_overpass_points(bbox, bbox["name"]))
        except Exception as exc:
            errors.append(str(exc))
        raw_points.extend(fetch_nominatim_points(bbox, bbox["name"]))

    raw_points.extend(fetch_nominatim_priority_sites())

    try:
        raw_points.extend(fetch_google_places_points())
    except Exception as exc:
        errors.append(f"Google Places gagal: {exc}")

    raw_csv = output_dir / "tps_jogja_ngaglik_raw.csv"
    clean_csv = output_dir / "tps_jogja_ngaglik_clean.csv"

    raw_df = pd.DataFrame(
        [
            {
                "name": p.name,
                "lat": p.lat,
                "lon": p.lon,
                "amenity": p.amenity,
                "object_type": p.object_type,
                "object_id": p.object_id,
                "source": p.source,
            }
            for p in raw_points
        ]
    )
    raw_df.to_csv(raw_csv, index=False)

    clean_df = clean_and_enrich(raw_points)
    clean_df.to_csv(clean_csv, index=False)

    print(f"Raw rows: {len(raw_df)}")
    print(f"Clean rows: {len(clean_df)}")
    print(f"Saved raw CSV: {raw_csv}")
    print(f"Saved clean CSV: {clean_csv}")

    if len(clean_df) == 0:
        raise RuntimeError(
            "Tidak ada titik valid hasil scraping. Coba ulang saat jaringan stabil atau set GOOGLE_MAPS_API_KEY untuk Google Places API."
        )

    if errors:
        print("Catatan endpoint:")
        for err in errors:
            print(f"- {err}")

    return raw_csv, clean_csv


if __name__ == "__main__":
    run_scrape(Path("data"))
