import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import normaltest, mannwhitneyu
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import logging
import time
import os

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_COLS      = ["Quantity", "UnitPrice", "Revenue", "TrueProfit"]
CATEGORICAL_COLS  = ["Country", "StockCode", "Description"]
FLAG_COLS         = ["Is_return", "Is_cancelled"]
TIME_COL          = "InvoiceDate"
REVENUE_COL       = "Revenue"
CUSTOMER_COL      = "Customer ID"

# thresholds
SKEW_MODERATE     = 0.5     # abs skew above this → moderate, consider sqrt
SKEW_HIGH         = 1.0     # abs skew above this → high, log1p recommended
CORR_HIGH         = 0.85    # abs correlation above this → flag as redundant
VIF_MODERATE      = 5.0     # VIF above this → investigate
VIF_HIGH          = 10.0    # VIF above this → drop feature from model
IMBALANCE_THRESH  = 0.15    # minority class below this % → imbalanced
IQR_MULTIPLIER    = 1.5     # standard IQR fence multiplier
OUTLIER_CAP_PCT   = 0.99    # percentile used for capping ML features
CHART_DPI         = 120
CHART_SAVE_PATH   = "reports/output/statistical_checks"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — section printer
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    """Print a clearly visible section header to stdout."""
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def _subsection(title: str) -> None:
    print(f"\n  ── {title} ──")


def _ok(msg: str)   -> None: print(f"    ✓  {msg}")
def _warn(msg: str) -> None: print(f"    ⚠  {msg}")
def _flag(msg: str) -> None: print(f"    ✗  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — DUPLICATES
# ─────────────────────────────────────────────────────────────────────────────

# def _check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Removes exact duplicate rows (every column identical).
#     Returns the deduplicated DataFrame.

#     Why first:
#         Duplicates inflate every statistic that follows — means, counts,
#         skewness, correlation. One duplicate bulk order row doubles its
#         Revenue contribution and pulls every downstream check off.
#     """
#     _section("CHECK 1 — DUPLICATES")

#     n_before  = len(df)
#     n_dupes   = df.duplicated().sum()
#     pct       = n_dupes / n_before * 100

#     print(f"\n    Rows before : {n_before:,}")
#     print(f"    Duplicates  : {n_dupes:,}  ({pct:.2f}%)")

#     if n_dupes == 0:
#         _ok("No duplicate rows found.")
#     else:
#         _warn(f"{n_dupes:,} duplicate rows detected — dropping.")
#         df = df.drop_duplicates().reset_index(drop=True)
#         _ok(f"Rows after deduplication: {len(df):,}")

#     logger.info(f"[1] Duplicates — removed {n_dupes:,} rows.")
#     return df


# # ─────────────────────────────────────────────────────────────────────────────
# # CHECK 2 — DATA TYPES
# # ─────────────────────────────────────────────────────────────────────────────

# def _check_dtypes(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Verifies and corrects column dtypes.
#     InvoiceDate must be datetime; numeric columns must not be object.

#     Why second:
#         Skewness, correlation, and outlier detection all call numeric
#         operations. If Revenue is stored as object (string), every
#         numeric check silently skips it or crashes.
#     """
#     _section("CHECK 2 — DATA TYPES")

#     _subsection("Current dtypes")
#     for col in df.columns:
#         print(f"    {col:<30} {str(df[col].dtype):<15}")

#     # fix InvoiceDate
#     if TIME_COL in df.columns:
#         if df[TIME_COL].dtype != "datetime64[ns]":
#             _warn(f"{TIME_COL} is {df[TIME_COL].dtype} — converting to datetime.")
#             df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
#             _ok(f"{TIME_COL} converted to datetime64.")
#         else:
#             _ok(f"{TIME_COL} is already datetime64.")

#     # check numeric columns are not stored as object
#     _subsection("Numeric column dtype validation")
#     for col in NUMERIC_COLS:
#         if col not in df.columns:
#             _warn(f"{col} not found in DataFrame — skipping.")
#             continue
#         if df[col].dtype == object:
#             _flag(f"{col} is stored as object — attempting numeric coercion.")
#             df[col] = pd.to_numeric(df[col], errors="coerce")
#             _ok(f"{col} coerced to {df[col].dtype}.")
#         else:
#             _ok(f"{col} dtype: {df[col].dtype}")

#     logger.info("[2] Data types validated and corrected.")
#     return df


# # ─────────────────────────────────────────────────────────────────────────────
# # CHECK 3 — MISSING VALUES
# # ─────────────────────────────────────────────────────────────────────────────

# def _check_missing(df: pd.DataFrame) -> None:
#     """
#     Reports null counts and percentages per column.
#     Does NOT drop anything — dropping decisions are in the cleaning function.
#     Documents what remains so downstream checks are not surprised by gaps.

#     Why third:
#         Outlier and skewness calculations on columns with significant nulls
#         produce biased statistics. Knowing the null picture before those
#         checks tells you how much to trust the results.
#     """
#     _section("CHECK 3 — MISSING VALUES")

#     null_counts = df.isnull().sum()
#     null_pct    = null_counts / len(df) * 100
#     summary     = pd.DataFrame({
#         "Null Count" : null_counts,
#         "Null %"     : null_pct.round(2)
#     }).query("`Null Count` > 0").sort_values("Null %", ascending=False)

#     if summary.empty:
#         _ok("No null values in any column.")
#     else:
#         print(f"\n{summary.to_string()}\n")
#         for col, row in summary.iterrows():
#             if row["Null %"] > 20:
#                 _flag(f"{col}: {row['Null %']:.1f}% null — verify handling in cleaning step.")
#             elif row["Null %"] > 5:
#                 _warn(f"{col}: {row['Null %']:.1f}% null — investigate.")
#             else:
#                 _ok(f"{col}: {row['Null %']:.1f}% null — acceptable.")

#     logger.info(f"[3] Missing values — {len(summary)} columns with nulls.")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — OUTLIERS (IQR)
# ─────────────────────────────────────────────────────────────────────────────

def _check_outliers(df: pd.DataFrame, save_charts: bool = True) -> dict:
    """
    Detects outliers in numeric columns using the IQR fence method.
    Prints counts, revenue impact, and saves boxplots.
    Returns a dict of {column: (lower_fence, upper_fence, cap_value)}
    for use in the ML feature capping step.

    Why fourth (before skewness):
        Outliers inflate skewness. Checking skewness on data that still
        has extreme bulk-order rows gives a misleadingly high skew value.
        Understand the outlier picture before interpreting skewness.

    Action guide:
        Genuine bulk order  → keep in revenue reports, tag as 'bulk'
        Price entry error   → cap or drop, document
        ML feature input    → always cap at 99th percentile
    """
    _section("CHECK 4 — OUTLIERS  (IQR method)")

    fences = {}

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue

        _subsection(col)
        series    = df[col].dropna()
        Q1        = series.quantile(0.25)
        Q3        = series.quantile(0.75)
        IQR       = Q3 - Q1
        lower     = Q1 - IQR_MULTIPLIER * IQR
        upper     = Q3 + IQR_MULTIPLIER * IQR
        cap_value = series.quantile(OUTLIER_CAP_PCT)

        outlier_mask  = (series < lower) | (series > upper)
        n_out         = outlier_mask.sum()
        pct_out       = n_out / len(series) * 100

        print(f"    Q1={Q1:.2f}  Q3={Q3:.2f}  IQR={IQR:.2f}")
        print(f"    Fences: lower={lower:.2f}  upper={upper:.2f}")
        print(f"    99th percentile cap (for ML): {cap_value:.2f}")
        print(f"    Outliers: {n_out:,}  ({pct_out:.2f}%)")

        if n_out == 0:
            _ok(f"No outliers in {col}.")
        else:
            # show the top 5 most extreme rows for human review
            extreme = (df.loc[series[outlier_mask].nlargest(5).index,
                               ["Invoice", "StockCode", "Description",
                                "Quantity", "UnitPrice", "Revenue"]]
                       if all(c in df.columns
                              for c in ["Invoice","StockCode","Description"])
                       else df.loc[series[outlier_mask].nlargest(5).index,
                                   [col]])
            print(f"\n    Top 5 highest outlier rows:\n{extreme.to_string()}\n")

            if col == REVENUE_COL:
                out_rev = series[outlier_mask].sum()
                total   = series.sum()
                _warn(f"Outlier {col} = £{out_rev:,.2f}  "
                      f"({out_rev/total*100:.1f}% of total revenue)")
                _warn("Investigate before capping — may be genuine wholesale orders.")
            else:
                _warn(f"{n_out:,} outliers. Cap at {cap_value:.2f} for ML features.")

        fences[col] = (lower, upper, cap_value)
        logger.info(f"[4] {col} — {n_out:,} outliers ({pct_out:.2f}%).")

    # boxplot grid
    if save_charts:
        os.makedirs(CHART_SAVE_PATH, exist_ok=True)
        fig, axes = plt.subplots(1, len(NUMERIC_COLS),
                                 figsize=(5 * len(NUMERIC_COLS), 5))
        for ax, col in zip(axes, NUMERIC_COLS):
            if col in df.columns:
                df[col].dropna().pipe(
                    lambda s: ax.boxplot(s, vert=True, patch_artist=True,
                                         boxprops=dict(facecolor="#AED6F1")))
                ax.set_title(col, fontsize=11)
                ax.set_xlabel("")
        plt.suptitle("Outlier Boxplots — Online Retail II", fontsize=13)
        plt.tight_layout()
        path = f"{CHART_SAVE_PATH}/outlier_boxplots.png"
        plt.savefig(path, dpi=CHART_DPI)
        plt.close()
        _ok(f"Boxplots saved → {path}")

    return fences


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5 — SKEWNESS
# ─────────────────────────────────────────────────────────────────────────────

def _check_skewness(df: pd.DataFrame, save_charts: bool = True) -> dict:
    """
    Measures skewness of all numeric columns.
    Recommends and applies log1p transform to copies — originals untouched.
    Returns {column: skew_value} for logging.

    Why log1p and not log:
        log(0) = -inf — crashes on any zero-quantity or zero-price row.
        log1p(x) = log(1 + x) — handles zeros safely.

    Why keep originals:
        Log-transformed Revenue means nothing to a stakeholder.
        Transform copies are for ML models only.
        Reports and dashboards always use original columns.
    """
    _section("CHECK 5 — SKEWNESS")

    skew_results = {}

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue

        skew_val = df[col].skew()
        skew_results[col] = skew_val

        print(f"\n    {col}:")
        print(f"      Skewness = {skew_val:.4f}")

        if abs(skew_val) <= SKEW_MODERATE:
            _ok("Approximately symmetric — no transform needed.")
            action = "none"

        elif abs(skew_val) <= SKEW_HIGH:
            _warn("Moderate skew — square-root transform recommended.")
            transformed = np.sqrt(df[col].clip(lower=0))
            df[f"{col}_sqrt"] = transformed
            new_skew = transformed.skew()
            _ok(f"Sqrt transform applied → {col}_sqrt  "
                f"(new skew: {new_skew:.4f})")
            action = "sqrt"

        else:
            _flag("High skew — log1p transform strongly recommended.")
            transformed = np.log1p(df[col].clip(lower=0))
            df[f"{col}_log"] = transformed
            new_skew = transformed.skew()
            _ok(f"Log1p transform applied → {col}_log  "
                f"(new skew: {new_skew:.4f})")
            action = "log1p"

        logger.info(f"[5] {col} — skew={skew_val:.4f}, action={action}.")

    # histogram grid — original vs transformed
    if save_charts:
        os.makedirs(CHART_SAVE_PATH, exist_ok=True)
        cols_to_plot = NUMERIC_COLS + [f"{c}_log" for c in NUMERIC_COLS
                                        if f"{c}_log" in df.columns]
        n = len(cols_to_plot)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, cols_to_plot):
            if col in df.columns:
                df[col].dropna().hist(bins=60, ax=ax, color="#2E86C1",
                                       edgecolor="white", linewidth=0.3)
                skew = df[col].skew()
                ax.set_title(f"{col}\nskew={skew:.2f}", fontsize=10)
        plt.suptitle("Skewness — Distributions", fontsize=13)
        plt.tight_layout()
        path = f"{CHART_SAVE_PATH}/skewness_histograms.png"
        plt.savefig(path, dpi=CHART_DPI)
        plt.close()
        _ok(f"Histograms saved → {path}")

    return skew_results


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6 — NORMALITY
# ─────────────────────────────────────────────────────────────────────────────

def _check_normality(df: pd.DataFrame, save_charts: bool = True) -> None:
    """
    Runs D'Agostino K² normality test on all numeric columns.
    Shapiro-Wilk is skipped — unreliable above n=5,000 and this dataset
    has ~500k rows where even trivial non-normality is statistically significant.

    D'Agostino K² tests both skewness and kurtosis simultaneously.
    p > 0.05 → fail to reject normality (distribution could be normal).
    p < 0.05 → reject normality (almost certain for retail revenue data).

    Action when non-normal:
        Comparing two groups  → Mann-Whitney U (not t-test)
        Comparing 3+ groups   → Kruskal-Wallis (not ANOVA)
        Regression residuals  → check Q-Q plot of residuals, not raw data
        KMeans / Prophet      → normality not required — no action needed
    """
    _section("CHECK 6 — NORMALITY  (D'Agostino K² test)")

    print("\n    Note: With ~500k rows, Shapiro-Wilk is unreliable.")
    print("    Using D'Agostino K² — tests skewness + kurtosis jointly.")
    print("    Retail revenue is almost never normally distributed.")
    print("    Non-normal → use non-parametric tests throughout.\n")

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        # sample if very large — test is valid on samples
        sample = series.sample(min(len(series), 50_000), random_state=42)

        stat, p = normaltest(sample)
        print(f"    {col}:")
        print(f"      D'Agostino K²: stat={stat:.4f}, p={p:.6f}")

        if p > 0.05:
            _ok("Fail to reject normality — parametric tests usable.")
        else:
            _flag("Normality rejected (p < 0.05).")
            print("      → Use Mann-Whitney U for two-group comparisons.")
            print("      → Use Kruskal-Wallis for multi-group comparisons.")
            print("      → Log-transform before regression if needed.")

        logger.info(f"[6] {col} — normality p={p:.6f}.")

    # Q-Q plots
    if save_charts:
        os.makedirs(CHART_SAVE_PATH, exist_ok=True)
        fig, axes = plt.subplots(1, len(NUMERIC_COLS),
                                 figsize=(5 * len(NUMERIC_COLS), 4))
        if len(NUMERIC_COLS) == 1:
            axes = [axes]
        for ax, col in zip(axes, NUMERIC_COLS):
            if col in df.columns:
                sample = df[col].dropna().sample(
                    min(len(df[col].dropna()), 10_000), random_state=42)
                (osm, osr), (slope, intercept, r) = stats.probplot(sample)
                ax.plot(osm, osr, "o", alpha=0.3, markersize=2,
                        color="#2E86C1", label="Data")
                ax.plot(osm, slope * np.array(osm) + intercept,
                        "r-", linewidth=1.5, label="Normal line")
                ax.set_title(f"{col} — Q-Q Plot", fontsize=10)
                ax.legend(fontsize=8)
        plt.suptitle("Normality — Q-Q Plots", fontsize=13)
        plt.tight_layout()
        path = f"{CHART_SAVE_PATH}/normality_qq.png"
        plt.savefig(path, dpi=CHART_DPI)
        plt.close()
        _ok(f"Q-Q plots saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 7 — CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def _check_correlation(df: pd.DataFrame, save_charts: bool = True) -> None:
    """
    Pearson correlation matrix on all numeric columns.
    Flags pairs above CORR_HIGH threshold as potentially redundant.

    Why this matters:
        Revenue = Quantity × UnitPrice by construction.
        All three will be highly correlated.
        Using Revenue AND Quantity as features in KMeans or regression
        double-counts the same information — inflating its importance.
        Correlation check tells you which features to drop before ML.

    Note: Correlation is pairwise. For full multicollinearity
    (one column predicted from a combination of others), see VIF below.
    """
    _section("CHECK 7 — CORRELATION")

    numeric_df = df[NUMERIC_COLS].dropna()
    corr       = numeric_df.corr(method="pearson")

    print(f"\n{corr.round(3).to_string()}\n")

    _subsection("High correlation pairs (abs > {:.2f})".format(CORR_HIGH))
    found_high = False
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if i >= j:
                continue
            r = corr.loc[c1, c2]
            if abs(r) > CORR_HIGH:
                _flag(f"{c1} vs {c2}: r={r:.3f} — drop one before ML.")
                print(f"      Reason: If r > {CORR_HIGH}, both columns carry")
                print(f"      near-identical signal. Include Revenue only")
                print(f"      OR Quantity only in model features, not both.")
                found_high = True

    if not found_high:
        _ok(f"No pairs above r={CORR_HIGH} threshold.")

    logger.info("[7] Correlation check complete.")

    # heatmap
    if save_charts:
        os.makedirs(CHART_SAVE_PATH, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, square=True, linewidths=0.5,
                    annot_kws={"size": 11}, ax=ax)
        ax.set_title("Pearson Correlation Matrix", fontsize=13)
        plt.tight_layout()
        path = f"{CHART_SAVE_PATH}/correlation_heatmap.png"
        plt.savefig(path, dpi=CHART_DPI)
        plt.close()
        _ok(f"Heatmap saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# BONUS CHECKS — ML-only (run conditionally)
# ─────────────────────────────────────────────────────────────────────────────

def _check_class_imbalance(df: pd.DataFrame) -> None:
    """
    Check binary flag columns for class imbalance.
    Only relevant before classification model training (churn, return prediction).
    """
    _section("BONUS CHECK A — CLASS IMBALANCE  (classification only)")
    print("    Relevant only when training churn or return classifiers.\n")

    for col in FLAG_COLS:
        if col not in df.columns:
            _warn(f"{col} not in DataFrame — skipping.")
            continue
        ratio     = df[col].mean()
        n_pos     = df[col].sum()
        n_neg     = len(df) - n_pos
        print(f"    {col}:")
        print(f"      Positive (1): {n_pos:,}  ({ratio:.1%})")
        print(f"      Negative (0): {n_neg:,}  ({1-ratio:.1%})")
        if ratio < IMBALANCE_THRESH:
            _flag(f"Imbalanced — minority class is {ratio:.1%}.")
            print("      → Use class_weight='balanced' in classifier.")
            print("      → Or: SMOTE oversampling via imbalanced-learn.")
        else:
            _ok(f"Acceptable balance — {ratio:.1%} positive class.")

    logger.info("[A] Class imbalance check complete.")


def _check_multicollinearity(df: pd.DataFrame) -> None:
    """
    Variance Inflation Factor for numeric columns.
    Only relevant before linear/logistic regression.
    VIF > 10 → severe multicollinearity → drop that feature.
    """
    _section("BONUS CHECK B — MULTICOLLINEARITY / VIF  (regression only)")
    print("    Relevant only when training linear or logistic regression.\n")

    features = df[NUMERIC_COLS].dropna()
    if len(features.columns) < 2:
        _warn("Need at least 2 numeric columns for VIF.")
        return

    vif_data = pd.DataFrame({
        "Feature": features.columns,
        "VIF": [variance_inflation_factor(features.values, i)
                for i in range(features.shape[1])]
    }).sort_values("VIF", ascending=False)

    print(f"\n{vif_data.to_string(index=False)}\n")
    for _, row in vif_data.iterrows():
        if row["VIF"] > VIF_HIGH:
            _flag(f"{row['Feature']}: VIF={row['VIF']:.2f} — "
                  "severe multicollinearity, drop from regression.")
        elif row["VIF"] > VIF_MODERATE:
            _warn(f"{row['Feature']}: VIF={row['VIF']:.2f} — moderate, investigate.")
        else:
            _ok(f"{row['Feature']}: VIF={row['VIF']:.2f} — acceptable.")

    logger.info("[B] VIF check complete.")


def _check_stationarity(df: pd.DataFrame) -> None:
    """
    Augmented Dickey-Fuller (ADF) test on monthly revenue series.
    Only relevant before Prophet or ARIMA forecasting.
    p < 0.05 → stationary (no unit root) → series is predictable.
    p > 0.05 → non-stationary → has trend → Prophet handles automatically.
    """
    _section("BONUS CHECK C — STATIONARITY / ADF  (forecasting only)")
    print("    Relevant only before Prophet or ARIMA forecasting.")
    print("    Prophet handles non-stationarity automatically —")
    print("    this check is for documentation, not intervention.\n")

    if "YearMonth" not in df.columns or REVENUE_COL not in df.columns:
        _warn("YearMonth or Revenue column not found — skipping ADF.")
        return

    monthly = (df.groupby("YearMonth")[REVENUE_COL]
               .sum()
               .sort_index())

    if len(monthly) < 12:
        _warn(f"Only {len(monthly)} monthly periods — "
              "ADF unreliable below 12 observations.")
        return

    result   = adfuller(monthly, autolag="AIC")
    adf_stat = result[0]
    p_val    = result[1]
    n_lags   = result[2]
    n_obs    = result[3]
    crit     = result[4]

    print(f"    ADF Statistic : {adf_stat:.4f}")
    print(f"    p-value       : {p_val:.6f}")
    print(f"    Lags used     : {n_lags}")
    print(f"    Observations  : {n_obs}")
    print(f"    Critical values:")
    for level, val in crit.items():
        print(f"      {level}: {val:.4f}")

    if p_val < 0.05:
        _ok("Stationary (p < 0.05) — series is predictable as-is.")
    else:
        _warn("Non-stationary (p ≥ 0.05) — trend or seasonality present.")
        print("    → Prophet accounts for this automatically.")
        print("    → Document: 'Revenue series is non-stationary;")
        print("       Prophet's additive/multiplicative decomposition")
        print("       handles trend and seasonality explicitly.'")

    logger.info(f"[C] ADF test — stat={adf_stat:.4f}, p={p_val:.6f}.")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_statistical_checks(
    df: pd.DataFrame,
    save_charts: bool = True,
    run_ml_checks: bool = False
) -> pd.DataFrame:
    """
    Run all statistical validation checks on the cleaned Online Retail II
    DataFrame, in the correct order.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned DataFrame from merged_table_Cleaning().
        Must contain: Quantity, UnitPrice, Revenue, InvoiceDate,
        Is_return, Is_cancelled, Customer ID.

    save_charts : bool, default True
        If True, saves PNG charts to reports/output/statistical_checks/.
        Set False for CI pipelines or headless environments.

    run_ml_checks : bool, default False
        If True, runs the three bonus checks:
            A — Class imbalance (for classification models)
            B — VIF / multicollinearity (for regression models)
            C — ADF stationarity (for time-series forecasting)
        These are not needed for EDA and business questions —
        only enable when preparing for ML model training.

    Returns
    -------
    pd.DataFrame
        The input DataFrame after:
            - Duplicate rows removed
            - Data types corrected
            - Log/sqrt transform columns appended (e.g. Revenue_log)
        All other checks are diagnostic only — no data is modified.

    Order rationale
    ---------------
    1. Duplicates    — must go first; inflate every stat that follows
    2. Data types    — numeric ops fail silently on wrong dtypes
    3. Missing       — know null picture before computing statistics
    4. Outliers      — inflate skewness; understand before interpreting it
    5. Skewness      — drives transform decision for ML features
    6. Normality     — determines parametric vs non-parametric test choice
    7. Correlation   — identifies redundant features before ML

    Usage
    -----
    from src.stats_checks import run_statistical_checks

    cleaned_df  = merged_table_Cleaning(merged_df)
    validated_df = run_statistical_checks(cleaned_df, save_charts=True)

    # For ML preparation:
    validated_df = run_statistical_checks(
        cleaned_df, save_charts=True, run_ml_checks=True)
    """
    overall_start = time.time()

    print("\n" + "█" * 60)
    print("  STATISTICAL VALIDATION REPORT — Online Retail II")
    print("█" * 60)
    print(f"\n  Input shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  save_charts  : {save_charts}")
    print(f"  run_ml_checks: {run_ml_checks}")

    # ── 7 CORE CHECKS ────────────────────────────────────────────────
    df = _check_duplicates(df)          # 1 — modifies df (drops dupes)
    df = _check_dtypes(df)              # 2 — modifies df (fixes dtypes)
    _check_missing(df)                  # 3 — diagnostic only
    _check_outliers(df, save_charts)    # 4 — diagnostic only
    df = _check_skewness(df, save_charts)  # 5 — adds *_log/*_sqrt cols
    _check_normality(df, save_charts)   # 6 — diagnostic only
    _check_correlation(df, save_charts) # 7 — diagnostic only

    # ── BONUS ML CHECKS (optional) ───────────────────────────────────
    if run_ml_checks:
        _check_class_imbalance(df)      # A — classification prep
        _check_multicollinearity(df)    # B — regression prep
        _check_stationarity(df)         # C — forecasting prep

    # ── SUMMARY ──────────────────────────────────────────────────────
    elapsed = time.time() - overall_start
    _section("VALIDATION COMPLETE")
    print(f"\n  Output shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  New columns  : "
          f"{[c for c in df.columns if c.endswith(('_log','_sqrt'))]}")
    print(f"  Time elapsed : {elapsed:.2f}s")
    if save_charts:
        print(f"  Charts saved : {CHART_SAVE_PATH}/")

    logger.info(
        f"Statistical validation complete in {elapsed:.2f}s. "
        f"Output: {df.shape[0]:,} rows × {df.shape[1]} columns."
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    pd.set_option("display.width", os.get_terminal_size().columns)
    pd.set_option("display.max_columns", None)

    print("Running stats_checks.py in standalone mode.")
    print("In production, call run_statistical_checks(cleaned_df) from pipeline.py")
    print("\nExample usage:")
    print("  from src.stats_checks import run_statistical_checks")
    print("  validated_df = run_statistical_checks(cleaned_df)")
    print("  validated_df = run_statistical_checks(cleaned_df, run_ml_checks=True)")