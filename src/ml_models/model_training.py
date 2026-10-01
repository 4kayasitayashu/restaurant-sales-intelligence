import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "daily_demand_forecasting.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "Orders"

TEST_DAYS = 54

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n[1/8] Loading forecasting dataset...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["Date"]
    )

    df = df.sort_values(
        [
            "Date",
            "Outlet"
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

    print(
        f"Date range : "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    return df


# ============================================================
# FEATURE SELECTION
# ============================================================

def prepare_features(df):

    print(
        "\n[2/8] Preparing ML features..."
    )

    df = df.copy()

    feature_columns = [

        "Outlet",

        "DayOfWeek",
        "Month",
        "Day",
        "WeekOfYear",
        "WeekOfMonth",
        "Quarter",
        "IsWeekend",

        "Orders_Lag_1",
        "Orders_Lag_7",
        "Orders_Lag_14",
        "Orders_Lag_28",

        "Orders_RollingMean_7",
        "Orders_RollingMean_14",
        "Orders_RollingMean_28"
    ]

    X = df[
        feature_columns
    ].copy()

    y = df[
        TARGET
    ].copy()

    print(
        f"Features : {len(feature_columns)}"
    )

    print(
        "\nFeatures used:"
    )

    for feature in feature_columns:

        print(
            f"  - {feature}"
        )

    return (
        X,
        y,
        feature_columns
    )


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

def chronological_split(
    df,
    X,
    y
):

    print(
        "\n[3/8] Creating chronological train/test split..."
    )

    unique_dates = (
        pd.Series(
            df["Date"].unique()
        )
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    if len(unique_dates) <= TEST_DAYS:

        raise ValueError(
            "Not enough unique dates for "
            "the requested test period."
        )

    test_start_date = (
        unique_dates.iloc[
            -TEST_DAYS
        ]
    )

    train_mask = (
        df["Date"]
        < test_start_date
    )

    test_mask = (
        df["Date"]
        >= test_start_date
    )

    X_train = X.loc[
        train_mask
    ].copy()

    X_test = X.loc[
        test_mask
    ].copy()

    y_train = y.loc[
        train_mask
    ].copy()

    y_test = y.loc[
        test_mask
    ].copy()

    print(
        f"Train rows : {len(X_train):,}"
    )

    print(
        f"Test rows  : {len(X_test):,}"
    )

    print(
        "\nTrain period:"
    )

    print(
        f"{df.loc[train_mask, 'Date'].min().date()} "
        f"to "
        f"{df.loc[train_mask, 'Date'].max().date()}"
    )

    print(
        "\nTest period:"
    )

    print(
        f"{df.loc[test_mask, 'Date'].min().date()} "
        f"to "
        f"{df.loc[test_mask, 'Date'].max().date()}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_mask,
        test_mask
    )


# ============================================================
# METRICS
# ============================================================

def calculate_wape(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:

        return np.nan

    return (
        np.sum(
            np.abs(
                actual - predicted
            )
        )
        / denominator
        * 100
    )


# ============================================================

def calculate_smape(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    mask = (
        denominator != 0
    )

    if not np.any(mask):

        return np.nan

    return (
        np.mean(
            2
            * np.abs(
                predicted[mask]
                - actual[mask]
            )
            / denominator[mask]
        )
        * 100
    )


# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    wape = calculate_wape(
        actual,
        predicted
    )

    smape = calculate_smape(
        actual,
        predicted
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_Percent": wape,
        "SMAPE_Percent": smape
    }


# ============================================================
# WEEKLY NAIVE BASELINE
# ============================================================

def run_baseline(
    df,
    test_mask
):

    print(
        "\n[4/8] Evaluating weekly-naive baseline..."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We create the actual and predicted values from the
    # SAME dataframe so their indexes/lengths always match.
    #
    # Prediction = demand from the same outlet 7 days earlier.
    # --------------------------------------------------------

    baseline_data = df.loc[
        test_mask,
        [
            "Date",
            "Outlet",
            "Orders",
            "Orders_Lag_7"
        ]
    ].copy()

    baseline_data = baseline_data.dropna(
        subset=[
            "Orders",
            "Orders_Lag_7"
        ]
    )

    actual = (
        baseline_data[
            "Orders"
        ]
        .to_numpy(
            dtype=float
        )
    )

    baseline_predictions = (
        baseline_data[
            "Orders_Lag_7"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if len(actual) == 0:

        raise ValueError(
            "Weekly-naive baseline has no "
            "valid test observations."
        )

    metrics = calculate_metrics(
        actual,
        baseline_predictions
    )

    print(
        f"\nBaseline observations: "
        f"{len(actual):,}"
    )

    print(
        "\nWeekly Naive Baseline:"
    )

    print(
        f"MAE   : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE  : {metrics['RMSE']:.4f}"
    )

    print(
        f"WAPE  : {metrics['WAPE_Percent']:.2f}%"
    )

    print(
        f"sMAPE : {metrics['SMAPE_Percent']:.2f}%"
    )

    return (
        metrics,
        baseline_data
    )


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor():

    numeric_features = [

        "DayOfWeek",
        "Month",
        "Day",
        "WeekOfYear",
        "WeekOfMonth",
        "Quarter",
        "IsWeekend",

        "Orders_Lag_1",
        "Orders_Lag_7",
        "Orders_Lag_14",
        "Orders_Lag_28",

        "Orders_RollingMean_7",
        "Orders_RollingMean_14",
        "Orders_RollingMean_28"
    ]

    categorical_features = [
        "Outlet"
    ]

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features
            ),

            (
                "categorical",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_models():

    models = {

        "LinearRegression":
            LinearRegression(),

        "RandomForest":
            RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "HistGradientBoosting":
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=RANDOM_STATE
            )
    }

    return models


# ============================================================
# TRAIN ML MODELS
# ============================================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    print(
        "\n[5/8] Training ML models..."
    )

    models = create_models()

    results = []

    trained_models = {}

    predictions = {}

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        # Create a fresh preprocessor for every model.
        # This avoids accidentally sharing fitted state.
        preprocessor = (
            create_preprocessor()
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        prediction = (
            pipeline
            .predict(
                X_test
            )
        )

        # Orders cannot be negative.
        prediction = np.maximum(
            prediction,
            0
        )

        metrics = calculate_metrics(
            y_test,
            prediction
        )

        results.append(
            {
                "Model": model_name,
                **metrics
            }
        )

        trained_models[
            model_name
        ] = pipeline

        predictions[
            model_name
        ] = prediction

        print(
            f"MAE   : {metrics['MAE']:.4f}"
        )

        print(
            f"RMSE  : {metrics['RMSE']:.4f}"
        )

        print(
            f"WAPE  : {metrics['WAPE_Percent']:.2f}%"
        )

        print(
            f"sMAPE : {metrics['SMAPE_Percent']:.2f}%"
        )

    results_df = pd.DataFrame(
        results
    )

    return (
        results_df,
        trained_models,
        predictions
    )


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

def save_results(
    results,
    trained_models,
    predictions,
    df,
    test_mask,
    baseline_data
):

    print(
        "\n[6/8] Saving model results..."
    )

    # --------------------------------------------------------
    # Sort only ML models for selecting the best ML model.
    # --------------------------------------------------------

    ml_results = (
        results[
            results["Model"]
            != "WeeklyNaiveBaseline"
        ]
        .sort_values(
            "MAE"
        )
        .reset_index(
            drop=True
        )
    )

    if ml_results.empty:

        raise ValueError(
            "No ML model results available."
        )

    best_model_name = (
        ml_results
        .iloc[0]["Model"]
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    comparison_file = (
        REPORT_DIR
        / "forecast_model_comparison.csv"
    )

    final_results = (
        results
        .sort_values(
            "MAE"
        )
        .reset_index(
            drop=True
        )
    )

    final_results.to_csv(
        comparison_file,
        index=False
    )

    print(
        "\nModel comparison saved:"
    )

    print(
        comparison_file
    )

    # --------------------------------------------------------
    # Create prediction dataframe
    # --------------------------------------------------------

    prediction_df = df.loc[
        test_mask,
        [
            "Date",
            "Outlet",
            "Orders"
        ]
    ].copy()

    prediction_df = (
        prediction_df
        .rename(
            columns={
                "Orders":
                    "ActualOrders"
            }
        )
    )

    # --------------------------------------------------------
    # Add weekly baseline prediction
    # --------------------------------------------------------

    prediction_df = prediction_df.merge(
        baseline_data[
            [
                "Date",
                "Outlet",
                "Orders_Lag_7"
            ]
        ],
        on=[
            "Date",
            "Outlet"
        ],
        how="left"
    )

    prediction_df = (
        prediction_df
        .rename(
            columns={
                "Orders_Lag_7":
                    "WeeklyNaive_Prediction"
            }
        )
    )

    # --------------------------------------------------------
    # Add ML predictions
    # --------------------------------------------------------

    for model_name, prediction in predictions.items():

        # predictions correspond exactly to X_test,
        # which follows the test_mask row order.
        prediction_df[
            f"{model_name}_Prediction"
        ] = prediction

    prediction_file = (
        REPORT_DIR
        / "forecast_test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False
    )

    print(
        "\nTest predictions saved:"
    )

    print(
        prediction_file
    )

    # --------------------------------------------------------
    # Save best ML model
    # --------------------------------------------------------

    best_model = (
        trained_models[
            best_model_name
        ]
    )

    best_model_file = (
        MODEL_DIR
        / "restaurant_demand_model.joblib"
    )

    joblib.dump(
        best_model,
        best_model_file
    )

    print(
        "\nBest ML model saved:"
    )

    print(
        best_model_file
    )

    print(
        f"Best ML model: "
        f"{best_model_name}"
    )

    return (
        final_results,
        best_model_name,
        prediction_df
    )


# ============================================================
# CREATE SUMMARY
# ============================================================

def create_summary(
    results,
    best_model_name,
    df,
    test_mask
):

    print(
        "\n[7/8] Creating model summary..."
    )

    best_row = (
        results[
            results["Model"]
            == best_model_name
        ]
        .iloc[0]
    )

    summary_file = (
        REPORT_DIR
        / "forecast_model_summary.txt"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "RESTAURANT DEMAND FORECASTING\n"
        )

        file.write(
            "========================================\n\n"
        )

        file.write(
            "Target:\n"
        )

        file.write(
            "Daily Orders per Outlet\n\n"
        )

        file.write(
            "Dataset:\n"
        )

        file.write(
            f"{DATA_FILE}\n\n"
        )

        file.write(
            "Forecasting approach:\n"
        )

        file.write(
            "Leakage-safe supervised machine learning\n\n"
        )

        file.write(
            "Train/Test strategy:\n"
        )

        file.write(
            "Chronological split\n\n"
        )

        file.write(
            "Test period:\n"
        )

        file.write(
            f"{df.loc[test_mask, 'Date'].min().date()} "
            f"to "
            f"{df.loc[test_mask, 'Date'].max().date()}\n\n"
        )

        file.write(
            "Model comparison:\n\n"
        )

        file.write(
            results.to_string(
                index=False
            )
        )

        file.write(
            "\n\n"
        )

        file.write(
            "Selected ML model:\n"
        )

        file.write(
            f"{best_model_name}\n\n"
        )

        file.write(
            "Selected model metrics:\n"
        )

        file.write(
            f"MAE: "
            f"{best_row['MAE']:.4f}\n"
        )

        file.write(
            f"RMSE: "
            f"{best_row['RMSE']:.4f}\n"
        )

        file.write(
            f"WAPE: "
            f"{best_row['WAPE_Percent']:.2f}%\n"
        )

        file.write(
            f"sMAPE: "
            f"{best_row['SMAPE_Percent']:.2f}%\n"
        )

    print(
        "\nSummary saved:"
    )

    print(
        summary_file
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "RESTAURANT DEMAND FORECASTING"
    )

    print(
        "LEAKAGE-SAFE ML MODEL TRAINING"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Features
    # --------------------------------------------------------

    (
        X,
        y,
        feature_columns
    ) = prepare_features(
        df
    )

    # --------------------------------------------------------
    # 3. Chronological split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_mask,
        test_mask
    ) = chronological_split(
        df,
        X,
        y
    )

    # --------------------------------------------------------
    # 4. Baseline
    # --------------------------------------------------------

    (
        baseline_metrics,
        baseline_data
    ) = run_baseline(
        df,
        test_mask
    )

    baseline_row = {
        "Model":
            "WeeklyNaiveBaseline",
        **baseline_metrics
    }

    # --------------------------------------------------------
    # 5. Train ML models
    # --------------------------------------------------------

    (
        model_results,
        trained_models,
        predictions
    ) = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # Add baseline to comparison
    # --------------------------------------------------------

    results = pd.concat(
        [
            pd.DataFrame(
                [baseline_row]
            ),
            model_results
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # 6. Save results
    # --------------------------------------------------------

    (
        final_results,
        best_model_name,
        prediction_df
    ) = save_results(
        results,
        trained_models,
        predictions,
        df,
        test_mask,
        baseline_data
    )

    # --------------------------------------------------------
    # 7. Create summary
    # --------------------------------------------------------

    create_summary(
        final_results,
        best_model_name,
        df,
        test_mask
    )

    # --------------------------------------------------------
    # 8. Final output
    # --------------------------------------------------------

    print(
        "\n[8/8] Final model comparison..."
    )

    print()

    print(
        final_results.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "MODEL TRAINING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSelected ML model: "
        f"{best_model_name}"
    )

    print(
        "\nGenerated files:"
    )

    print(
        "  - reports\\forecast_model_comparison.csv"
    )

    print(
        "  - reports\\forecast_test_predictions.csv"
    )

    print(
        "  - reports\\forecast_model_summary.txt"
    )

    print(
        "  - models\\restaurant_demand_model.joblib"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()