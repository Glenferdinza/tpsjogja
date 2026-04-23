from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import folium
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DATA_PATH = Path("data/tps_jogja_ngaglik_clean.csv")
OUT_ROOT = Path("ppt_assets_per_slide")
TARGET_NODES = 15

NGAGLIK_BBOX = {
    "min_lat": -7.76,
    "max_lat": -7.68,
    "min_lon": 110.34,
    "max_lon": 110.44,
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = np.radians([lat1, lat2])
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def in_bbox(df: pd.DataFrame, bbox: Dict[str, float]) -> pd.DataFrame:
    return df[
        (df["lat"].between(bbox["min_lat"], bbox["max_lat"]))
        & (df["lon"].between(bbox["min_lon"], bbox["max_lon"]))
    ].copy()


def build_edges_with_features(nodes_df: pd.DataFrame, detour_factor: float = 1.2, alpha: float = 0.1, beta: float = 0.1) -> pd.DataFrame:
    rows = []
    for i in range(len(nodes_df)):
        for j in range(i + 1, len(nodes_df)):
            a = nodes_df.iloc[i]
            b = nodes_df.iloc[j]

            d_geo = haversine_km(float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"]))
            d_road = d_geo * detour_factor

            avg_waste = (float(a["waste_ton_per_day"]) + float(b["waste_ton_per_day"])) / 2
            avg_access = (float(a["access_score"]) + float(b["access_score"])) / 2
            cost = d_road * (1 + alpha * (avg_waste / 10) - beta * avg_access)

            rows.append(
                {
                    "u": a["node"],
                    "v": b["node"],
                    "u_name": a["name"],
                    "v_name": b["name"],
                    "distance_geo_km": d_geo,
                    "distance_road_km": d_road,
                    "avg_waste_ton": avg_waste,
                    "avg_access_score": avg_access,
                    "cost": cost,
                }
            )

    return pd.DataFrame(rows).sort_values("cost").reset_index(drop=True)


def compute_mst_from_edge_df(edge_df: pd.DataFrame, weight_col: str = "cost") -> Tuple[nx.Graph, float, float]:
    g = nx.Graph()
    for _, r in edge_df.iterrows():
        g.add_edge(r["u"], r["v"], weight=float(r[weight_col]))

    mst_graph = nx.minimum_spanning_tree(g, algorithm="kruskal", weight="weight")
    total_weight = sum(d["weight"] for _, _, d in mst_graph.edges(data=True))
    max_edge = max(d["weight"] for _, _, d in mst_graph.edges(data=True))
    return mst_graph, total_weight, max_edge


def save_df_table_image(df: pd.DataFrame, out_path: Path, title: str, max_rows: int = 20) -> None:
    show_df = df.head(max_rows).copy()
    fig_height = 1.2 + 0.35 * (len(show_df) + 1)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    table = ax.table(
        cellText=show_df.values,
        colLabels=show_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def ensure_slide_dirs() -> Dict[str, Path]:
    dirs = {
        "01": OUT_ROOT / "01_judul",
        "02": OUT_ROOT / "02_latar_belakang",
        "03": OUT_ROOT / "03_dataset_dan_sumber",
        "04": OUT_ROOT / "04_data_preparation",
        "05": OUT_ROOT / "05_metode_kruskal_dan_tuning",
        "06": OUT_ROOT / "06_hasil_data_model",
        "07": OUT_ROOT / "07_tabel_edge_kandidat",
        "08": OUT_ROOT / "08_tabel_hasil_optimasi",
        "09": OUT_ROOT / "09_visualisasi_graf",
        "10": OUT_ROOT / "10_visualisasi_peta",
        "11": OUT_ROOT / "11_kesimpulan",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    clean_df = pd.read_csv(DATA_PATH)
    clean_df.columns = [c.lower() for c in clean_df.columns]
    clean_df = clean_df.dropna(subset=["lat", "lon"]).copy()
    clean_df["name"] = clean_df["name"].astype(str).str.strip()
    clean_df = clean_df.drop_duplicates(subset=["name", "lat", "lon"]).reset_index(drop=True)

    ngaglik_probe = in_bbox(clean_df, NGAGLIK_BBOX)
    if len(ngaglik_probe) > 0:
        center_lat = float(ngaglik_probe["lat"].mean())
        center_lon = float(ngaglik_probe["lon"].mean())
    else:
        center_lat = float(clean_df["lat"].mean())
        center_lon = float(clean_df["lon"].mean())

    clean_df["dist_to_ngaglik_center_km"] = clean_df.apply(
        lambda r: haversine_km(float(r["lat"]), float(r["lon"]), center_lat, center_lon), axis=1
    )

    ngaglik_df = in_bbox(clean_df, NGAGLIK_BBOX)
    sleman_df = clean_df[clean_df.get("district_estimate", "").astype(str).str.lower().eq("sleman")].copy()

    priority_df = pd.concat([ngaglik_df, sleman_df], ignore_index=True)
    priority_df = priority_df.drop_duplicates(subset=["name", "lat", "lon"])
    priority_df = priority_df.sort_values(["dist_to_ngaglik_center_km", "name"], ascending=[True, True])

    if len(priority_df) < TARGET_NODES:
        chosen_keys = set(zip(priority_df["name"], priority_df["lat"], priority_df["lon"]))
        spill_df = clean_df[~clean_df.apply(lambda r: (r["name"], r["lat"], r["lon"]) in chosen_keys, axis=1)].copy()
        spill_df = spill_df.sort_values(["dist_to_ngaglik_center_km", "name"], ascending=[True, True])
        tps_df = pd.concat([priority_df, spill_df], ignore_index=True).head(TARGET_NODES).copy()
    else:
        tps_df = priority_df.head(TARGET_NODES).copy()

    tps_df = tps_df.reset_index(drop=True)
    tps_df["node"] = [f"V{i+1}" for i in range(len(tps_df))]

    rng = np.random.default_rng(42)
    tps_df["waste_ton_per_day"] = rng.uniform(1.5, 8.0, size=len(tps_df)).round(2)
    tps_df["access_score"] = rng.uniform(0.45, 0.95, size=len(tps_df)).round(3)

    grid_detour = [1.10, 1.15, 1.20, 1.25]
    grid_alpha = [0.00, 0.05, 0.10, 0.15, 0.20]
    grid_beta = [0.00, 0.05, 0.10, 0.15, 0.20]
    grid_gamma = [0.00, 0.05, 0.10]

    tuning_rows = []
    for detour in grid_detour:
        for alpha in grid_alpha:
            for beta in grid_beta:
                edge_df = build_edges_with_features(tps_df, detour_factor=detour, alpha=alpha, beta=beta)
                mst_g, mst_total, mst_max = compute_mst_from_edge_df(edge_df)
                for gamma in grid_gamma:
                    objective = mst_total + gamma * mst_max
                    tuning_rows.append(
                        {
                            "detour_factor": detour,
                            "alpha_volume_penalty": alpha,
                            "beta_access_bonus": beta,
                            "gamma_max_edge_penalty": gamma,
                            "mst_total_cost": mst_total,
                            "mst_max_edge": mst_max,
                            "objective": objective,
                        }
                    )

    tuning_df = pd.DataFrame(tuning_rows).sort_values("objective").reset_index(drop=True)
    best = tuning_df.iloc[0]

    final_edges_df = build_edges_with_features(
        tps_df,
        detour_factor=float(best["detour_factor"]),
        alpha=float(best["alpha_volume_penalty"]),
        beta=float(best["beta_access_bonus"]),
    )
    final_mst_graph, final_mst_total, final_mst_max = compute_mst_from_edge_df(final_edges_df)

    mst_edge_keys = {tuple(sorted((u, v))) for u, v in final_mst_graph.edges()}
    edge_table_df = final_edges_df.copy()
    edge_table_df["status_kruskal"] = edge_table_df.apply(
        lambda r: "Diterima" if tuple(sorted((r["u"], r["v"]))) in mst_edge_keys else "Ditolak (Siklus)",
        axis=1,
    )
    optimized_edges_df = edge_table_df[edge_table_df["status_kruskal"] == "Diterima"].copy()

    summary_df = pd.DataFrame(
        [
            ["Jumlah titik clean", len(clean_df)],
            ["Jumlah node model", len(tps_df)],
            ["Jumlah edge kandidat", len(edge_table_df)],
            ["Jumlah edge MST", len(optimized_edges_df)],
            ["Total biaya MST", round(final_mst_total, 4)],
            ["Edge terpanjang MST", round(final_mst_max, 4)],
        ],
        columns=["metrik", "nilai"],
    )

    dirs = ensure_slide_dirs()

    # Slide 03, 06, 07, 08
    save_df_table_image(
        tps_df[["node", "name", "lat", "lon", "amenity", "district_estimate"]],
        dirs["03"] / "dataset_table.png",
        "Dataset Titik TPS Yang Dipakai",
        max_rows=20,
    )

    save_df_table_image(summary_df, dirs["06"] / "summary_metrics.png", "Ringkasan Hasil Model", max_rows=10)

    save_df_table_image(
        edge_table_df[["u", "v", "distance_road_km", "cost", "status_kruskal"]],
        dirs["07"] / "edge_table_kandidat.png",
        "Tabel Edge Kandidat",
        max_rows=30,
    )

    save_df_table_image(
        optimized_edges_df[["u", "v", "u_name", "v_name", "distance_road_km", "cost"]],
        dirs["08"] / "edge_table_mst.png",
        "Tabel Hasil Optimasi MST",
        max_rows=30,
    )

    # Slide 05 tuning + kruskal notes
    save_df_table_image(
        tuning_df.head(12),
        dirs["05"] / "tuning_top_results.png",
        "Top Hyperparameter Tuning",
        max_rows=12,
    )

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")
    ax.set_title("Flow Kruskal + Tuning", fontsize=14, fontweight="bold")
    flow_text = (
        "Scraping Data -> Cleaning -> Feature Engineering -> Bangun Edge Berbobot\n"
        "-> Sort Edge (ascending) -> Cek Siklus -> Pilih Edge\n"
        "-> MST Kruskal -> Hitung Objective -> Grid Search Hyperparameter"
    )
    ax.text(0.5, 0.5, flow_text, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#eef7f2", edgecolor="#1b9e77"))
    fig.tight_layout()
    fig.savefig(dirs["05"] / "metode_flow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Slide 09 graph comparison
    node_pos = {r["node"]: (float(r["lon"]), float(r["lat"])) for _, r in tps_df.iterrows()}
    full_graph = nx.Graph()
    for _, r in final_edges_df.iterrows():
        full_graph.add_edge(r["u"], r["v"], weight=float(r["cost"]))

    fig = plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    nx.draw(
        full_graph,
        pos=node_pos,
        with_labels=True,
        node_size=800,
        node_color="#fee8c8",
        font_size=8,
        width=0.9,
        edge_color="#666666",
    )
    plt.title("Graf Kandidat Edge")

    plt.subplot(1, 2, 2)
    nx.draw(
        final_mst_graph,
        pos=node_pos,
        with_labels=True,
        node_size=820,
        node_color="#ccebc5",
        font_size=8,
        width=2.0,
        edge_color="#1b9e77",
    )
    plt.title("Graf Optimasi MST Kruskal")
    fig.tight_layout()
    fig.savefig(dirs["09"] / "graf_kandidat_vs_mst.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Slide 10 map outputs
    center_lat = float(tps_df["lat"].mean())
    center_lon = float(tps_df["lon"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")

    for _, r in tps_df.iterrows():
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=5,
            color="#2c7fb8",
            fill=True,
            fill_opacity=0.9,
            popup=f"{r['node']} - {r['name']}",
        ).add_to(m)

    for _, r in optimized_edges_df.iterrows():
        p1 = tps_df.loc[tps_df["node"] == r["u"], ["lat", "lon"]].iloc[0].tolist()
        p2 = tps_df.loc[tps_df["node"] == r["v"], ["lat", "lon"]].iloc[0].tolist()
        folium.PolyLine(locations=[p1, p2], color="#d7301f", weight=3, opacity=0.85).add_to(m)

    m.save(str(dirs["10"] / "peta_mst_interaktif.html"))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(tps_df["lon"], tps_df["lat"], s=60, c="#2c7fb8", label="Node TPS")
    for _, r in optimized_edges_df.iterrows():
        p1 = tps_df.loc[tps_df["node"] == r["u"], ["lat", "lon"]].iloc[0]
        p2 = tps_df.loc[tps_df["node"] == r["v"], ["lat", "lon"]].iloc[0]
        ax.plot([p1["lon"], p2["lon"]], [p1["lat"], p2["lat"]], color="#d7301f", linewidth=1.8)
    for _, r in tps_df.iterrows():
        ax.text(float(r["lon"]) + 0.001, float(r["lat"]) + 0.001, str(r["node"]), fontsize=8)
    ax.set_title("Visualisasi Peta Statik MST")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(dirs["10"] / "peta_mst_statik.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Slide 01 and 11 reuse key visuals
    (dirs["01"] / "cover_visual.png").write_bytes((dirs["10"] / "peta_mst_statik.png").read_bytes())
    (dirs["11"] / "closing_visual_mst.png").write_bytes((dirs["09"] / "graf_kandidat_vs_mst.png").read_bytes())

    # Slide 02 and 04 textual visuals
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    ax.set_title("Latar Belakang (Ilustrasi Teks)", fontsize=14, fontweight="bold")
    ax.text(
        0.5,
        0.55,
        "Biaya transportasi sampah tinggi -> perlu rute minimum\nGunakan MST Kruskal untuk mengurangi total biaya koneksi antar TPS",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff3e0", edgecolor="#ef6c00"),
    )
    fig.tight_layout()
    fig.savefig(dirs["02"] / "latar_belakang_visual.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    ax.set_title("Data Preparation Flow", fontsize=14, fontweight="bold")
    ax.text(
        0.5,
        0.55,
        "Raw CSV -> Cleaning -> Filtering Ngaglik/Sleman -> Feature Engineering -> Graph Modeling",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#e8f5e9", edgecolor="#2e7d32"),
    )
    fig.tight_layout()
    fig.savefig(dirs["04"] / "data_preparation_flow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    readme = OUT_ROOT / "README_SLIDE_ASSETS.txt"
    readme.write_text(
        "Folder aset per slide sudah dibuat.\n"
        "01_judul: cover_visual.png\n"
        "02_latar_belakang: latar_belakang_visual.png\n"
        "03_dataset_dan_sumber: dataset_table.png\n"
        "04_data_preparation: data_preparation_flow.png\n"
        "05_metode_kruskal_dan_tuning: metode_flow.png, tuning_top_results.png\n"
        "06_hasil_data_model: summary_metrics.png\n"
        "07_tabel_edge_kandidat: edge_table_kandidat.png\n"
        "08_tabel_hasil_optimasi: edge_table_mst.png\n"
        "09_visualisasi_graf: graf_kandidat_vs_mst.png\n"
        "10_visualisasi_peta: peta_mst_statik.png, peta_mst_interaktif.html\n"
        "11_kesimpulan: closing_visual_mst.png\n",
        encoding="utf-8",
    )

    print(f"Selesai export aset PPT ke folder: {OUT_ROOT}")


if __name__ == "__main__":
    main()
