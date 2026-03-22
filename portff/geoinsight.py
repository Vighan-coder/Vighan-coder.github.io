# ============================================================
#  GeoInsight — Geospatial Crime Analytics
#  Author : Vighan Raj Verma (@Vighan-coder)
#  GitHub : https://github.com/Vighan-coder/GeoInsight
# ============================================================
#
#  SETUP:
#    pip install pandas numpy matplotlib seaborn scikit-learn
#                folium h3 geopandas shapely
#
#  RUN:
#    python geoinsight.py
#
#  OUTPUT:
#    crime_map.html          — interactive Folium map
#    hotspot_map.html        — H3 hexagonal heatmap
#    cluster_results.png     — DBSCAN cluster plot
#    model_metrics.txt       — F1 / precision / recall
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import folium
import h3
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from folium.plugins import HeatMap, MarkerCluster


# ── City bounding boxes ─────────────────────────────────────
CITIES = {
    "Mumbai"   : (18.87, 19.28, 72.77, 73.05),
    "Delhi"    : (28.40, 28.88, 76.84, 77.35),
    "Bangalore": (12.83, 13.14, 77.46, 77.75),
    "Hyderabad": (17.28, 17.56, 78.33, 78.61),
    "Chennai"  : (12.90, 13.23, 80.15, 80.31),
}
CRIME_TYPES = ["Theft","Assault","Robbery","Vandalism","Fraud",
               "Burglary","Drug Offence","Harassment"]


# ════════════════════════════════════════════════════════════
#  1. SYNTHETIC CRIME DATA GENERATOR
# ════════════════════════════════════════════════════════════
def generate_crime_data(n_per_city=500, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []
    for city, (lat_min, lat_max, lon_min, lon_max) in CITIES.items():
        # Create 3–4 crime hotspot clusters per city
        n_clusters = np.random.randint(3, 5)
        centers    = [
            (np.random.uniform(lat_min, lat_max),
             np.random.uniform(lon_min, lon_max))
            for _ in range(n_clusters)
        ]
        for _ in range(n_per_city):
            cluster_id    = np.random.randint(0, n_clusters)
            clat, clon    = centers[cluster_id]
            lat  = np.random.normal(clat, 0.015)
            lon  = np.random.normal(clon, 0.015)
            lat  = np.clip(lat, lat_min, lat_max)
            lon  = np.clip(lon, lon_min, lon_max)
            hour = int(np.random.choice(
                range(24),
                p=np.array([1,1,1,1,1,1,2,3,4,5,5,5,5,5,5,5,5,5,6,6,5,4,3,2],
                           dtype=float) / 90.0
            ))
            crime_type = np.random.choice(CRIME_TYPES,
                p=[0.25,0.15,0.12,0.10,0.13,0.10,0.08,0.07])
            rows.append({
                "city"        : city,
                "latitude"    : round(lat, 6),
                "longitude"   : round(lon, 6),
                "crime_type"  : crime_type,
                "hour"        : hour,
                "day_of_week" : np.random.randint(0, 7),
                "month"       : np.random.randint(1, 13),
                "severity"    : np.random.choice(
                    ["Low","Medium","High"], p=[0.50,0.35,0.15]),
                "cluster_center": cluster_id,
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════
#  2. DBSCAN CLUSTERING
# ════════════════════════════════════════════════════════════
def run_dbscan(df: pd.DataFrame, eps_km=0.5, min_samples=8):
    """
    eps_km : radius in km (approx — 1 degree lat ≈ 111 km)
    """
    eps_deg = eps_km / 111.0
    coords  = df[["latitude","longitude"]].values
    scaler  = StandardScaler()
    coords_s = scaler.fit_transform(coords)

    db = DBSCAN(eps=eps_deg / (coords.std() + 1e-9),
                min_samples=min_samples,
                metric="euclidean")
    df["dbscan_cluster"] = db.fit_predict(coords_s)

    n_clusters = len(set(df["dbscan_cluster"])) - (1 if -1 in df["dbscan_cluster"].values else 0)
    n_noise    = (df["dbscan_cluster"] == -1).sum()
    print(f"[GeoInsight] DBSCAN → {n_clusters} clusters | {n_noise} noise points")
    return df


# ════════════════════════════════════════════════════════════
#  3. H3 HEXAGONAL BINNING
# ════════════════════════════════════════════════════════════
def add_h3_index(df: pd.DataFrame, resolution=8) -> pd.DataFrame:
    df["h3_index"] = df.apply(
        lambda r: h3.geo_to_h3(r["latitude"], r["longitude"], resolution), axis=1)
    return df


def get_h3_counts(df: pd.DataFrame):
    return df.groupby("h3_index").size().reset_index(name="count")


# ════════════════════════════════════════════════════════════
#  4. MAPS
# ════════════════════════════════════════════════════════════
def make_crime_map(df: pd.DataFrame, output="crime_map.html"):
    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=11,
                   tiles="CartoDB dark_matter")

    # Heatmap layer
    heat_data = df[["latitude","longitude"]].values.tolist()
    HeatMap(heat_data, radius=12, blur=15, min_opacity=0.4).add_to(m)

    # Cluster markers
    mc = MarkerCluster().add_to(m)
    color_map = {"Theft":"red","Assault":"orange","Robbery":"darkred",
                 "Vandalism":"blue","Fraud":"green","Burglary":"purple",
                 "Drug Offence":"darkblue","Harassment":"gray"}
    for _, row in df.sample(min(500, len(df))).iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4, color=color_map.get(row["crime_type"],"gray"),
            fill=True, fill_opacity=0.7,
            popup=f"{row['crime_type']} | {row['city']} | Hour:{row['hour']}"
        ).add_to(mc)

    m.save(output)
    print(f"[Saved] {output}")
    return m


def make_h3_map(df: pd.DataFrame, output="hotspot_map.html"):
    try:
        h3_df  = add_h3_index(df)
        counts = get_h3_counts(h3_df)
        center = [df["latitude"].mean(), df["longitude"].mean()]
        m = folium.Map(location=center, zoom_start=11,
                       tiles="CartoDB dark_matter")

        max_count = counts["count"].max()
        for _, row in counts.iterrows():
            boundary = h3.h3_to_geo_boundary(row["h3_index"], geo_json=True)
            norm     = row["count"] / max_count
            r, g, b  = int(255*norm), int(255*(1-norm)), 50
            color    = f"#{r:02x}{g:02x}{b:02x}"
            folium.Polygon(
                locations=[[c[1],c[0]] for c in boundary],
                color=color, fill=True, fill_color=color,
                fill_opacity=0.55,
                popup=f"Count: {row['count']}"
            ).add_to(m)

        m.save(output)
        print(f"[Saved] {output}")
    except Exception as e:
        print(f"[GeoInsight] H3 map skipped: {e}")


# ════════════════════════════════════════════════════════════
#  5. DBSCAN CLUSTER PLOT
# ════════════════════════════════════════════════════════════
def plot_clusters(df: pd.DataFrame, city="Mumbai"):
    sub = df[df["city"] == city].copy()
    sub = run_dbscan(sub, eps_km=0.3, min_samples=5)

    cmap   = plt.get_cmap("tab20")
    labels = sub["dbscan_cluster"].values
    colors = [cmap(l % 20) if l != -1 else (0.5,0.5,0.5,0.5) for l in labels]

    plt.figure(figsize=(8,6))
    plt.scatter(sub["longitude"], sub["latitude"],
                c=colors, s=8, alpha=0.7)
    plt.title(f"DBSCAN Crime Clusters — {city}")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig("cluster_results.png", dpi=150)
    plt.show()
    print("[Saved] cluster_results.png")


# ════════════════════════════════════════════════════════════
#  6. HOTSPOT PREDICTION MODEL
# ════════════════════════════════════════════════════════════
def build_prediction_model(df: pd.DataFrame):
    df = df.copy()
    le = LabelEncoder()
    df["city_enc"]        = le.fit_transform(df["city"])
    df["crime_type_enc"]  = le.fit_transform(df["crime_type"])
    df["severity_enc"]    = le.fit_transform(df["severity"])

    features = ["latitude","longitude","hour","day_of_week",
                "month","city_enc","crime_type_enc"]
    X = df[features]
    y = (df["severity"] == "High").astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.08, subsample=0.8, random_state=42)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    report = classification_report(y_te, y_pred,
                                   target_names=["Normal","Hotspot"])
    f1     = f1_score(y_te, y_pred)

    print("\n── Hotspot Prediction ─────────────────────────────")
    print(report)
    print(f"F1-Score: {f1:.4f}")

    with open("model_metrics.txt", "w") as fh:
        fh.write(report)
        fh.write(f"\nF1-Score: {f1:.4f}\n")
    print("[Saved] model_metrics.txt")

    # Feature importance
    importances = pd.Series(clf.feature_importances_, index=features)
    importances.sort_values().plot(kind="barh", color="#7cff67", figsize=(7,4))
    plt.title("Feature Importance — Hotspot Predictor")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("[Saved] feature_importance.png")
    return clf


# ════════════════════════════════════════════════════════════
#  7. CRIME TREND ANALYSIS
# ════════════════════════════════════════════════════════════
def plot_trends(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Hourly distribution
    df.groupby("hour").size().plot(ax=axes[0], color="#7cff67")
    axes[0].set_title("Crimes by Hour of Day")
    axes[0].set_xlabel("Hour"); axes[0].set_ylabel("Count")

    # Crime type by city
    pivot = pd.crosstab(df["city"], df["crime_type"])
    pivot.plot(kind="bar", ax=axes[1], colormap="tab10", legend=True)
    axes[1].set_title("Crime Type by City")
    axes[1].tick_params(axis="x", rotation=30)

    # Severity distribution
    df["severity"].value_counts().plot(
        kind="pie", ax=axes[2],
        colors=["#7cff67","#B19EEF","#ff6b6b"],
        autopct="%1.1f%%")
    axes[2].set_title("Severity Distribution")
    axes[2].set_ylabel("")

    plt.suptitle("GeoInsight — Crime Analytics Dashboard", fontsize=14)
    plt.tight_layout()
    plt.savefig("crime_trends.png", dpi=150)
    plt.show()
    print("[Saved] crime_trends.png")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[GeoInsight] Generating crime dataset …")
    df = generate_crime_data(n_per_city=600)
    print(f"  Total records : {len(df)}")
    print(f"  Cities        : {df['city'].unique().tolist()}")
    print(f"  Crime types   : {df['crime_type'].unique().tolist()}\n")

    # DBSCAN on full dataset
    df = run_dbscan(df)

    # Maps
    make_crime_map(df)
    make_h3_map(df)

    # City-level cluster plot
    plot_clusters(df, city="Mumbai")

    # Trend analysis
    plot_trends(df)

    # Predictive model
    build_prediction_model(df)

    print("\n[GeoInsight] Done! Open crime_map.html in your browser.")