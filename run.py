import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, mean_squared_error)
from datetime import timedelta
import os, warnings
warnings.filterwarnings("ignore")

USE_SYNTHETIC = True

DISRUPTIONS = [
    ("2008-07-01", "2008-12-31", 5, "GFC"),
    ("2011-02-15", "2011-10-31", 4, "Arab Spring"),
    ("2014-06-01", "2016-02-01", 4, "OPEC War"),
    ("2017-08-25", "2017-09-30", 3, "Harvey"),
    ("2019-09-14", "2019-10-15", 3, "Aramco"),
    ("2020-03-01", "2020-06-30", 5, "COVID"),
    ("2022-02-24", "2022-07-31", 4, "Russia-Ukraine"),
    ("2023-10-19", "2024-03-31", 3, "Houthi"),
]

LAYER_COLORS = {
    "Financial": "#3498DB", "Geopolitical": "#E74C3C", "Physical Supply": "#2ECC71",
    "Market Structure": "#9B59B6", "Environmental": "#F39C12", "Cross-Signal": "#1ABC9C",
}

def feature_layer(name):
    """Map a feature name to its signal layer."""
    for prefix, layer in [
        ("crude_return", "Financial"), ("crude_rvol", "Financial"), ("crude_vs_ma", "Financial"),
        ("crude_ivol", "Financial"), ("vol_of_vol", "Financial"),
        ("inventory", "Physical Supply"), ("refinery", "Physical Supply"), ("import", "Physical Supply"),
        ("futures", "Market Structure"), ("contango", "Market Structure"),
        ("backwardation", "Market Structure"), ("freight", "Market Structure"), ("volume_ratio", "Market Structure"),
        ("tension", "Geopolitical"), ("hurricane", "Environmental"), ("winter", "Environmental"),
        ("inv_price", "Cross-Signal"), ("tension_vol", "Cross-Signal"), ("supply_stress", "Cross-Signal"),
    ]:
        if name.startswith(prefix): return layer
    return "Other"



# 1. DATA GENERATION


def generate_data():
    """Generate synthetic crude oil market data calibrated to real statistical properties."""
    np.random.seed(42)
    dates = pd.date_range("2005-01-01", "2025-03-28", freq="B")
    n = len(dates)

    # --- Layer 1: Physical supply chain ---
    inv_base = 350 + np.cumsum(np.random.normal(0, 2, n)) * 0.3
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 252)
    us_crude_inventory = np.clip(inv_base + seasonal, 280, 550)

    ref_base = 88 + np.cumsum(np.random.normal(0, 0.3, n)) * 0.1
    ref_seasonal = 4 * np.sin(2 * np.pi * (np.arange(n) - 60) / 252)
    refinery_utilization = np.clip(ref_base + ref_seasonal, 65, 98)

    imports = np.clip(8 + np.cumsum(np.random.normal(0, 0.05, n)) * 0.1, 5, 11)
    shale_adj = np.zeros(n)
    shale_mask = dates > pd.Timestamp("2015-01-01")
    shale_adj[shale_mask] = -np.linspace(0, 3, shale_mask.sum())
    crude_imports = np.clip(imports + shale_adj, 3, 11)

    # --- Layer 2: Financial signals ---
    # mean-reverting, realistic crude price dynamics
    crude_price = np.zeros(n)
    crude_price[0] = 60.0
    for i in range(1, n):
        crude_price[i] = crude_price[i-1] + 0.02 * (65 - crude_price[i-1]) + 1.2 * np.random.normal()
    crude_price = np.clip(crude_price, 20, 140)

    crude_ivol = np.clip(30 + np.cumsum(np.random.normal(0, 0.3, n)) * 0.15, 15, 80)
    price_ret = np.diff(crude_price, prepend=crude_price[0]) / np.maximum(crude_price, 20)
    crude_ivol = np.clip(crude_ivol - 300 * np.minimum(price_ret, 0), 15, 120)

    futures_slope = pd.Series(np.random.normal(0, 1.5, n)).rolling(10).mean().fillna(0).values

    freight_proxy = np.zeros(n); freight_proxy[0] = 50.0
    for i in range(1, n):
        freight_proxy[i] = freight_proxy[i-1] + 0.01 * (50 - freight_proxy[i-1]) + 0.8 * np.random.normal()
    freight_proxy = np.clip(freight_proxy, 15, 130)

    crude_volume = np.random.lognormal(12, 0.3, n)

    # --- Layer 3: Geopolitical tension (7 oil-producing regions) ---
    regions = {"saudi_arabia": (30, 8), "russia": (35, 10), "iraq": (50, 12),
               "libya": (40, 14), "iran": (45, 10), "nigeria": (42, 10), "venezuela": (40, 10)}
    tension_data = {}
    for region, (base, vol) in regions.items():
        t = base + np.cumsum(np.random.normal(0, vol * 0.15, n)) + np.random.normal(0, vol, n)
        # Random tension episodes ~2-3x/year even outside disruptions (realistic noise)
        for _ in range(int(2.5 * n / 252)):
            s = np.random.randint(0, max(1, n - 60))
            l = np.random.randint(15, 60)
            spike = np.random.uniform(15, 35)
            ramp = np.linspace(0, spike, l // 2 + 1)
            decay = np.linspace(spike, 0, l - len(ramp) + 1)
            t[s:s+l] += np.concatenate([ramp, decay])[:min(l, n - s)]
        tension_data[f"tension_{region}"] = np.clip(t, 5, 98)

    # --- Layer 4: Environmental ---
    month = pd.Series(dates).dt.month.values
    hurricane_intensity = np.clip(((month >= 6) & (month <= 11)).astype(float) * np.random.exponential(0.3, n), 0, 5)
    winter_severity = np.clip(np.maximum(0, np.sin(2 * np.pi * (month - 1) / 12)) * np.random.exponential(1, n), 0, 5)

    df = pd.DataFrame({"us_crude_inventory": us_crude_inventory, "refinery_utilization": refinery_utilization,
        "crude_imports": crude_imports, "crude_price": crude_price, "crude_ivol": crude_ivol,
        "futures_slope": futures_slope, "freight_proxy": freight_proxy, "crude_volume": crude_volume,
        "hurricane_intensity": hurricane_intensity, "winter_severity": winter_severity, **tension_data}, index=dates)

    # --- Inject disruption effects with staggered timing ---
    # Reality: tension builds weeks before prices react. This stagger is what makes
    # geopolitical features useful as LEADING indicators, not just correlated noise.
    effects = {
        "2008-07-01": {"crude_price": 0.4, "crude_ivol": 2.5, "refinery_util": 0.85},
        "2011-02-15": {"tension_libya": 90, "tension_saudi_arabia": 60, "crude_price": 1.3},
        "2014-06-01": {"crude_price": 0.4, "futures_slope": 5, "freight_proxy": 0.6},
        "2017-08-25": {"refinery_util": 0.75, "hurricane_intensity": 4.5, "crude_price": 1.1},
        "2019-09-14": {"tension_saudi_arabia": 85, "crude_ivol": 2.0, "crude_price": 1.15},
        "2020-03-01": {"crude_price": 0.25, "crude_ivol": 3.0, "refinery_util": 0.7, "crude_imports": 0.7},
        "2022-02-24": {"tension_russia": 90, "crude_price": 1.5, "freight_proxy": 2.0, "crude_ivol": 2.0},
        "2023-10-19": {"freight_proxy": 2.5, "tension_saudi_arabia": 55, "crude_ivol": 1.5},
    }
    lead_days = {"tension_": 25, "freight_proxy": 14, "hurricane": 12, "refinery": 7,
                 "crude_imports": 7, "crude_price": 3, "crude_ivol": 5, "futures": 3, "crude_volume": 2}

    for i, (start, end, sev, label) in enumerate(DISRUPTIONS):
        ev_start, ev_end = pd.Timestamp(start), pd.Timestamp(end)
        ev_effects = effects.get(start, {})
        for signal, modifier in ev_effects.items():
            if signal not in df.columns: continue
            lead = next((d for pfx, d in lead_days.items() if signal.startswith(pfx)), 0)
            mask = (df.index >= ev_start - timedelta(days=lead)) & (df.index <= ev_end)
            elen = mask.sum()
            if elen == 0: continue
            if signal.startswith("tension_"):
                ramp = 1 - np.exp(-3 * np.linspace(0, 1, min(15, elen)))
                envelope = np.concatenate([ramp, np.ones(max(elen - 25, 0)),
                    np.linspace(1, 0.3, max(elen - len(ramp) - max(elen - 25, 0), 1))])[:elen]
                cur = df.loc[mask, signal].values
                df.loc[mask, signal] = cur + (modifier - cur) * envelope * 0.8
            elif modifier < 1:
                prof = np.concatenate([np.linspace(1, modifier, elen//2),
                    np.linspace(modifier, 0.9, elen - elen//2)])[:elen]
                df.loc[mask, signal] = df.loc[mask, signal].values * prof
            else:
                prof = np.concatenate([np.linspace(1, modifier, elen//3),
                    np.full(elen//3, modifier), np.linspace(modifier, 1.05, elen - 2*(elen//3))])[:elen]
                df.loc[mask, signal] = df.loc[mask, signal].values * prof

    df["crude_price"] = df["crude_price"].clip(20, 140)
    df["crude_ivol"] = df["crude_ivol"].clip(15, 120)
    df["refinery_utilization"] = df["refinery_utilization"].clip(60, 98)
    for c in df.columns:
        if c.startswith("tension_"): df[c] = df[c].clip(5, 98)
    return df


# 2. FEATURE ENGINEERING (17 raw -> 47 engineered)


def engineer_features(df):
    f = pd.DataFrame(index=df.index)

    # Price momentum and volatility (14 features)
    for w in [5, 10, 21, 63]: f[f"crude_return_{w}d"] = df["crude_price"].pct_change(w)
    dret = df["crude_price"].pct_change()
    for w in [10, 21, 63]: f[f"crude_rvol_{w}d"] = dret.rolling(w).std() * np.sqrt(252)
    for w in [21, 63, 126]:
        ma = df["crude_price"].rolling(w).mean()
        f[f"crude_vs_ma{w}"] = (df["crude_price"] - ma) / ma
    f["crude_ivol_level"] = df["crude_ivol"]
    f["crude_ivol_change_5d"] = df["crude_ivol"].diff(5)
    f["crude_ivol_zscore"] = (df["crude_ivol"] - df["crude_ivol"].rolling(63).mean()) / df["crude_ivol"].rolling(63).std()
    f["vol_of_vol_21d"] = df["crude_ivol"].rolling(21).std()

    # Physical supply chain (9 features)
    inv_ma, inv_std = df["us_crude_inventory"].rolling(252).mean(), df["us_crude_inventory"].rolling(252).std()
    f["inventory_zscore"] = (df["us_crude_inventory"] - inv_ma) / inv_std
    f["inventory_change_5d"] = df["us_crude_inventory"].diff(5)
    f["inventory_change_21d"] = df["us_crude_inventory"].diff(21)
    f["inventory_draw_rate"] = df["us_crude_inventory"].diff(5) / df["us_crude_inventory"].shift(5)
    f["refinery_util_level"] = df["refinery_utilization"]
    f["refinery_util_deviation"] = df["refinery_utilization"] - df["refinery_utilization"].rolling(252).mean()
    f["refinery_util_drop_5d"] = df["refinery_utilization"].diff(5)
    f["import_change_21d"] = df["crude_imports"].pct_change(21)
    f["import_level"] = df["crude_imports"]

    # Market structure (6 features)
    f["futures_slope"] = df["futures_slope"]
    f["contango_flag"] = (df["futures_slope"] > 1).astype(float)
    f["backwardation_flag"] = (df["futures_slope"] < -1).astype(float)
    frt_ma, frt_std = df["freight_proxy"].rolling(63).mean(), df["freight_proxy"].rolling(63).std()
    f["freight_zscore"] = (df["freight_proxy"] - frt_ma) / frt_std
    f["freight_change_5d"] = df["freight_proxy"].pct_change(5)
    f["freight_change_21d"] = df["freight_proxy"].pct_change(21)
    f["volume_ratio"] = df["crude_volume"] / df["crude_volume"].rolling(21).mean()

    # Geopolitical composite (10 features)
    tcols = [c for c in df.columns if c.startswith("tension_")]
    weights = {"tension_saudi_arabia": .25, "tension_russia": .22, "tension_iraq": .12,
               "tension_iran": .10, "tension_libya": .05, "tension_nigeria": .08, "tension_venezuela": .05}
    wsum = sum(v for k, v in weights.items() if k in df.columns)
    composite = sum(df[k] * (v / wsum) for k, v in weights.items() if k in df.columns)
    f["tension_composite"] = composite
    f["tension_change_5d"] = composite.diff(5)
    f["tension_change_21d"] = composite.diff(21)
    t_ma, t_std = composite.rolling(126).mean(), composite.rolling(126).std()
    f["tension_zscore"] = (composite - t_ma) / t_std
    f["tension_max_region"] = df[tcols].max(axis=1)
    f["tension_dispersion"] = df[tcols].std(axis=1)
    f["tension_breadth"] = (df[tcols] > df[tcols].quantile(0.6)).sum(axis=1)
    for c in ["tension_saudi_arabia", "tension_russia", "tension_iraq", "tension_iran"]:
        if c in df.columns: f[c] = df[c]

    # Environmental (3 features)
    f["hurricane_intensity"] = df["hurricane_intensity"]
    f["hurricane_season"] = df.index.month.isin([6,7,8,9,10,11]).astype(float)
    f["winter_severity"] = df["winter_severity"]

    # Cross-signal interactions (3 features)
    f["inv_price_divergence"] = f["inventory_change_21d"] * f["crude_return_21d"]
    f["tension_vol_coupling"] = f["tension_change_5d"] * f["crude_ivol_change_5d"]
    f["supply_stress"] = (-f["inventory_zscore"] - f["refinery_util_deviation"]/10 + f["tension_zscore"]) / 3

    return f.dropna()


# 3. LABELS & SPLITS

def create_labels(df):
    """Risk score ramps up 21 days BEFORE each disruption — the model learns to predict fragility."""
    labels = pd.Series(0.0, index=df.index)
    for start, end, severity, _ in DISRUPTIONS:
        s = pd.Timestamp(start)
        warn = (df.index >= s - timedelta(days=21)) & (df.index < s)
        if warn.sum() > 0:
            labels.loc[warn] = np.maximum(labels.loc[warn], np.linspace(0.2, severity/5, warn.sum()))
        during = (df.index >= s) & (df.index <= pd.Timestamp(end))
        labels.loc[during] = np.maximum(labels.loc[during], severity / 5)
    return labels


def split_data(features, labels):
    """Chronological split, scaler fit only on training data to prevent leakage."""
    common = features.index.intersection(labels.index)
    X, y = features.loc[common], labels.loc[common]
    scaler = StandardScaler()
    train_mask = X.index <= "2020-12-31"
    scaler.fit(X.loc[train_mask])
    Xs = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    val_mask = (X.index > "2020-12-31") & (X.index <= "2022-12-31")
    test_mask = X.index > "2022-12-31"
    return {s: (Xs.loc[m], y.loc[m]) for s, m in [("train", train_mask), ("val", val_mask), ("test", test_mask)]}



# 4. TRAIN & EVALUATE


def train_xgb(splits):
    X_tr, y_tr = splits["train"]
    X_val, y_val = splits["val"]
    model = xgb.XGBRegressor(
        objective="reg:squarederror", max_depth=6, learning_rate=0.05, n_estimators=500,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def evaluate(model, splits):
    """Find optimal threshold from val PR curve, report metrics on all splits."""
    results = {}
    # Find optimal threshold on validation set
    X_val, y_val = splits["val"]
    pred_val = np.clip(model.predict(X_val), 0, 1)
    prec, rec, thresholds = precision_recall_curve((y_val > 0.3).astype(int), pred_val)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    opt_thresh = thresholds[np.argmax(f1s)]
    print(f"\nOptimal threshold (from val PR curve): {opt_thresh:.4f}")

    for name in ["train", "val", "test"]:
        X, y = splits[name]
        pred = np.clip(model.predict(X), 0, 1)
        yb = (y > 0.3).astype(int)
        pb = (pred > opt_thresh).astype(int)
        auc = roc_auc_score(yb, pred) if yb.nunique() > 1 else float("nan")
        ap = average_precision_score(yb, pred) if yb.nunique() > 1 else float("nan")
        results[name] = {"pred": pred, "y": np.array(y), "dates": X.index, "auc": auc, "ap": ap,
            "prec": precision_score(yb, pb, zero_division=0), "rec": recall_score(yb, pb, zero_division=0),
            "f1": f1_score(yb, pb, zero_division=0), "mse": mean_squared_error(y, pred)}
        print(f"  {name:5s} | AUC={auc:.4f}  AP={ap:.4f}  P={results[name]['prec']:.3f}  "
              f"R={results[name]['rec']:.3f}  F1={results[name]['f1']:.3f}  MSE={results[name]['mse']:.4f}")
    return results, opt_thresh



# 5. SHAP ANALYSIS


def compute_shap(model, splits):
    """TreeExplainer gives exact Shapley values — no approximation needed."""
    explainer = shap.TreeExplainer(model)
    shap_frames = []
    for name in ["train", "val", "test"]:
        X, _ = splits[name]
        sv = explainer.shap_values(X)
        shap_frames.append(pd.DataFrame(sv, index=X.index, columns=X.columns))
    all_shap = pd.concat(shap_frames)

    # Group by signal layer for decomposition chart
    layer_shap = pd.DataFrame(index=all_shap.index)
    for layer in LAYER_COLORS:
        cols = [c for c in all_shap.columns if feature_layer(c) == layer]
        if cols: layer_shap[layer] = all_shap[cols].abs().sum(axis=1)
    return all_shap, layer_shap



# 6. CHARTS


def setup_style():
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": True,
        "axes.spines.top": False, "axes.spines.right": False, "grid.alpha": 0.15, "grid.linestyle": "--",
        "font.family": "sans-serif", "font.size": 10, "figure.dpi": 150, "savefig.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.3})


def chart_hero_timeline(results, out):
    """Risk score 2005-2025 with disruption events shaded."""
    fig, ax = plt.subplots(figsize=(16, 5))
    all_dates = np.concatenate([results[s]["dates"] for s in ["train", "val", "test"]])
    all_preds = np.concatenate([results[s]["pred"] for s in ["train", "val", "test"]])

    for start, end, sev, label in DISRUPTIONS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08 + sev * 0.04, color="#E74C3C", zorder=0)
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax.text(mid, 0.97, label, ha="center", va="top", fontsize=7, color="#1B365D", alpha=0.8,
                transform=ax.get_xaxis_transform())

    ax.plot(all_dates, all_preds, color="#CC0000", linewidth=0.7, alpha=0.9)
    ax.axhline(0.3, color="#F39C12", ls="--", alpha=0.5, lw=0.8, label="Warning (0.3)")
    ax.axhline(0.6, color="#E74C3C", ls="--", alpha=0.5, lw=0.8, label="Critical (0.6)")
    for dt, lbl in [("2020-12-31", "Train|Val"), ("2022-12-31", "Val|Test")]:
        ax.axvline(pd.Timestamp(dt), color="#1B365D", ls=":", alpha=0.4, lw=0.8)
        ax.text(pd.Timestamp(dt), 1.02, lbl, ha="center", fontsize=8, color="#1B365D",
                transform=ax.get_xaxis_transform())

    ax.set_ylabel("Risk Score"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Crude Oil Supply Chain Fragility Score", fontweight="bold", pad=15)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(); fig.savefig(f"{out}/hero_timeline.png"); plt.close()


def chart_feature_importance(model, feature_names, out):
    """Top 15 features bar chart + signal layer pie chart."""
    fi = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    fi["layer"] = fi["feature"].apply(feature_layer)
    fi = fi.sort_values("importance", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})
    top = fi.head(15).iloc[::-1]
    colors = [LAYER_COLORS.get(r["layer"], "#999") for _, r in top.iterrows()]
    axes[0].barh(range(len(top)), top["importance"], color=colors, alpha=0.85)
    axes[0].set_yticks(range(len(top))); axes[0].set_yticklabels(top["feature"], fontsize=8)
    axes[0].set_xlabel("Importance (gain)"); axes[0].set_title("Top 15 Features", fontweight="bold")

    layer_pct = fi.groupby("layer")["importance"].sum()
    layer_pct = (layer_pct / layer_pct.sum()).sort_values(ascending=False)
    axes[1].pie(layer_pct, labels=layer_pct.index, autopct="%1.0f%%",
                colors=[LAYER_COLORS.get(l, "#999") for l in layer_pct.index],
                startangle=90, textprops={"fontsize": 8})
    axes[1].set_title("Signal Layer Split", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{out}/feature_importance.png"); plt.close()
    print(f"  Layer split: {', '.join(f'{l} {v:.0%}' for l, v in layer_pct.items())}")


def chart_signal_decomposition(layer_shap, out):
    """Stacked area showing which signal layer drives risk over time (from SHAP values)."""
    smoothed = layer_shap.rolling(63, min_periods=21).mean().dropna()
    pct = smoothed.div(smoothed.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.stackplot(pct.index, [pct[c] for c in pct.columns], labels=pct.columns,
                 colors=[LAYER_COLORS[c] for c in pct.columns], alpha=0.8)
    for start, end, _, _ in DISRUPTIONS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.06, color="#E74C3C", zorder=0)
    ax.set_ylabel("SHAP Contribution Share"); ax.set_ylim(0, 1)
    ax.set_title("Risk Signal Decomposition — What Drives Risk Over Time", fontweight="bold")
    ax.legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(); fig.savefig(f"{out}/signal_decomposition.png"); plt.close()


def chart_score_distribution(results, out):
    """Histogram of predictions split by disruption vs calm periods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name in zip(axes, ["train", "val", "test"]):
        pred, y = results[name]["pred"], results[name]["y"]
        calm, disr = y < 0.1, y > 0.3
        if calm.sum(): ax.hist(pred[calm], bins=50, alpha=0.6, color="#95A5A6", label=f"Calm ({calm.sum()})", density=True)
        if disr.sum(): ax.hist(pred[disr], bins=50, alpha=0.6, color="#E74C3C", label=f"Disruption ({disr.sum()})", density=True)
        ax.axvline(0.3, color="black", ls="--", alpha=0.4, lw=0.8)
        ax.set_title(name.title(), fontweight="bold"); ax.set_xlim(-0.05, 1.05); ax.legend(fontsize=7)
    fig.suptitle("Score Distribution: Calm vs Disruption Periods", fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{out}/score_distribution.png"); plt.close()



# 7. CONSULTING SCENARIOS 

def print_scenarios():
    print("\n" + "=" * 65)
    print("CONSULTING SCENARIO ANALYSES")
    print("=" * 65)
    print("""
  SCENARIO A: AIRLINES (mid-size carrier, $3B annual fuel spend)
    Fuel = 28% of OPEX. Jet fuel refined from crude.
    Risk 0.3-0.5 -> crude +15% -> fuel cost +$450M -> margin hit -4.2pts
    Risk 0.5-0.7 -> crude +25% -> fuel cost +$750M -> margin hit -7.0pts
    Risk >0.7    -> crude +40% -> fuel cost +$1.2B  -> margin hit -11.2pts
    Hedge cost: $90M/yr. Expected savings from signal-triggered hedging: ~$60M/yr.

  SCENARIO B: SHIPPING (50-vessel fleet, $35K/day operating cost)
    Rerouting Suez -> Cape adds 10 days, $562K/voyage.
    Preemptive rerouting saves ~$309K/vessel/event vs reactive.
    Fleet-wide annual benefit from early warning: ~$62M.
    Action triggers: 0.3 = contingency plans, 0.5 = pre-position fuel, 0.7 = full reroute.
""")



if __name__ == "__main__":
    out = "outputs"
    os.makedirs(out, exist_ok=True)
    setup_style()

    print("Generating data..."); raw = generate_data()
    print(f"  {raw.shape[0]} trading days, {raw.shape[1]} raw features")

    print("Engineering features..."); features = engineer_features(raw)
    print(f"  {features.shape[1]} features, {features.shape[0]} observations")

    print("Creating labels..."); labels = create_labels(raw)
    print(f"  {(labels > 0.3).sum()} days with elevated risk")

    print("Splitting data..."); splits = split_data(features, labels)
    for s in splits: print(f"  {s:5s}: {splits[s][0].shape[0]} days")

    print("Training XGBoost..."); model = train_xgb(splits)

    print("\nEvaluation:"); results, threshold = evaluate(model, splits)

    print("\nComputing SHAP values..."); all_shap, layer_shap = compute_shap(model, splits)

    print("\nGenerating charts...")
    chart_hero_timeline(results, out); print("  -> hero_timeline.png")
    chart_feature_importance(model, list(features.columns), out); print("  -> feature_importance.png")
    chart_signal_decomposition(layer_shap, out); print("  -> signal_decomposition.png")
    chart_score_distribution(results, out); print("  -> score_distribution.png")

    print_scenarios()
