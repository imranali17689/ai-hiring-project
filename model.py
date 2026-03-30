import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def normalize_col_name(name: str) -> str:
    """Normalize column names for robust dataset loading."""
    return name.strip().lower().replace("-", "_").replace(" ", "")


def selection_rate(preds: np.ndarray, genders: pd.Series, label: str) -> float:
    """Compute selection rate = P(pred=1 | gender=label)."""
    mask = genders.astype(str).str.strip().values == label
    if mask.sum() == 0:
        return float("nan")
    return preds[mask].mean()


def main() -> None:
    data_path = "Adult.csv"

    if not os.path.exists(data_path):
        print(f"Error: could not find dataset file `{data_path}` in the repo root.")
        sys.exit(1)

    # Load dataset (from repo root)
    df = pd.read_csv(data_path)

    # Print raw column names first (easy verification)
    print("Dataset column names (raw):")
    print(list(df.columns))

    # Normalize column names: lowercase, strip spaces, hyphens -> underscores
    df.columns = [normalize_col_name(c) for c in df.columns]

    # Detect target column safely: income if present, else class
    if "income" in df.columns:
        target_col = "income"
    elif "class" in df.columns:
        target_col = "class"
    else:
        print(
            "Error: could not find target column.\n"
            "Expected `income` or `class` after normalization.\n"
            f"Available columns: {list(df.columns)}"
        )
        sys.exit(1)

    # Detect gender (sensitive attribute) safely: sex must exist
    if "sex" not in df.columns:
        print(
            "Error: missing gender column.\n"
            "Expected column `sex` after normalization.\n"
            f"Available columns: {list(df.columns)}"
        )
        sys.exit(1)
    gender_col = "sex"

    # Features (must exist)
    feature_cols = ["age", "education_num", "hours_per_week"]
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        print("Error: missing required feature columns after normalization:")
        print(missing_features)
        print("Available columns:")
        print(list(df.columns))
        sys.exit(1)

    # Clean/parse gender values
    df[gender_col] = df[gender_col].astype(str).str.strip()

    # Convert target to binary:
    # >50k or >50K => 1
    # <=50k or <=50K => 0
    target_raw = df[target_col].astype(str).str.strip()
    target_norm = target_raw.str.upper().str.replace(" ", "", regex=False)

    y = pd.Series(np.nan, index=df.index, dtype="float")
    is_pos = target_norm.str.startswith(">50K")
    is_neg = target_norm.str.startswith("<=50K") | target_norm.str.startswith("<50K")
    y.loc[is_pos] = 1
    y.loc[is_neg] = 0

    df[target_col] = y

    # Convert features to numeric (invalid values become NaN)
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    needed_cols = feature_cols + [target_col, gender_col]
    before_rows = len(df)

    # Drop rows with missing values in needed columns
    df_model = df.dropna(subset=needed_cols).copy()
    after_rows = len(df_model)

    if after_rows == 0:
        print(
            "Error: after cleaning, there are 0 rows left to train on.\n"
            "This usually means many rows have missing/invalid values in the needed columns."
        )
        sys.exit(1)

    X = df_model[feature_cols]
    y = df_model[target_col].astype(int)
    genders = df_model[gender_col]

    # Split train/test
    X_train, X_test, y_train, y_test, genders_train, genders_test = train_test_split(
        X, y, genders, test_size=0.3, random_state=42
    )

    # Baseline model
    baseline_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)

    male_rate = selection_rate(baseline_preds, genders_test, "Male")
    female_rate = selection_rate(baseline_preds, genders_test, "Female")

    print("\nBASELINE MODEL")
    print(f"Target column: `{target_col}`")
    print(f"Feature columns: {feature_cols}")
    print(f"Dataset rows before dropna: {before_rows}")
    print(f"Dataset rows after dropna:  {after_rows}")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print(f"Male selection rate:   {male_rate:.4f}")
    print(f"Female selection rate: {female_rate:.4f}")

    # Simple fairness-adjusted model:
    # Reduce the favored group's positive predictions slightly, then round back to integers.
    adjusted_preds_float = baseline_preds.astype(float).copy()

    # Determine favored group based on baseline selection rates
    if np.isnan(male_rate) and np.isnan(female_rate):
        favored_group = "Male"
    elif np.isnan(male_rate):
        favored_group = "Female"
    elif np.isnan(female_rate):
        favored_group = "Male"
    else:
        favored_group = "Male" if male_rate > female_rate else "Female"

    # Reduce a small fraction of favored-group positives (deterministic with random_state=42)
    reduction_fraction = 0.2
    pos_mask_favored = (genders_test.astype(str).str.strip().values == favored_group) & (baseline_preds == 1)
    favored_pos_indices = np.where(pos_mask_favored)[0]

    n_reduced = 0
    rng = np.random.default_rng(42)
    if len(favored_pos_indices) > 0:
        n_reduced = int(np.ceil(reduction_fraction * len(favored_pos_indices)))
        n_reduced = min(n_reduced, len(favored_pos_indices))
        reduce_indices = rng.choice(favored_pos_indices, size=n_reduced, replace=False)

        # Reduce from 1.0 -> 0.1 so that rounding turns them into 0
        adjusted_preds_float[reduce_indices] = 0.1

    fairness_adjusted_preds = np.rint(adjusted_preds_float).astype(int)
    fairness_accuracy = accuracy_score(y_test, fairness_adjusted_preds)
    male_rate_fair = selection_rate(fairness_adjusted_preds, genders_test, "Male")
    female_rate_fair = selection_rate(fairness_adjusted_preds, genders_test, "Female")

    print("\nFAIRNESS-ADJUSTED MODEL (simple demo)")
    print(f"Favored group (higher baseline selection rate): {favored_group}")
    print(f"Reduced {n_reduced} favored positive predictions (fraction={reduction_fraction})")
    print(f"Fairness-adjusted accuracy: {fairness_accuracy:.4f}")
    print(f"Male selection rate (fair):   {male_rate_fair:.4f}")
    print(f"Female selection rate (fair): {female_rate_fair:.4f}")

    # Ensure results folder exists
    os.makedirs("results", exist_ok=True)

    # Configure Matplotlib to use a writable cache directory.
    # This prevents crashes in restricted environments where `~/.cache` isn't writable.
    cache_dir = os.path.abspath(os.path.join("results", ".cache"))
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)

    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for saving plots to files
    import matplotlib.pyplot as plt

    # Plot male vs female selection rates for baseline vs fairness-adjusted
    labels = ["Male", "Female"]
    baseline_rates = [male_rate, female_rate]
    fair_rates = [male_rate_fair, female_rate_fair]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, baseline_rates, width, label="Baseline")
    plt.bar(x + width / 2, fair_rates, width, label="Fairness-adjusted")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Selection rate (positive prediction)")
    plt.title("Selection Rate Comparison")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join("results", "selection_rate_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Save text summary
    summary_path = os.path.join("results", "results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("AI Hiring Fairness Project - Results Summary\n")
        f.write("=" * 48 + "\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Raw rows: {before_rows}\n")
        f.write(f"Rows after dropna for needed columns: {after_rows}\n")
        f.write(f"Target column: {target_col}\n")
        f.write(f"Gender column: sex\n")
        f.write(f"Feature columns: {feature_cols}\n\n")

        f.write("Baseline model (DecisionTreeClassifier, max_depth=3)\n")
        f.write("-" * 48 + "\n")
        f.write(f"Accuracy: {baseline_accuracy:.4f}\n")
        f.write(f"Male selection rate:   {male_rate:.4f}\n")
        f.write(f"Female selection rate: {female_rate:.4f}\n\n")

        f.write("Fairness-adjusted model (simple demo)\n")
        f.write("-" * 48 + "\n")
        f.write(f"Favored group: {favored_group}\n")
        f.write(f"Reduction fraction: {reduction_fraction}\n")
        f.write(f"Reduced favored positives: {n_reduced}\n")
        f.write(f"Accuracy: {fairness_accuracy:.4f}\n")
        f.write(f"Male selection rate:   {male_rate_fair:.4f}\n")
        f.write(f"Female selection rate: {female_rate_fair:.4f}\n\n")

        f.write(f"Plot saved to: {plot_path}\n")
        f.write(f"Summary saved to: {summary_path}\n")

    print("\nSaved outputs:")
    print(f"- Plot:   {plot_path}")
    print(f"- Summary:{summary_path}")


if __name__ == "__main__":
    main()