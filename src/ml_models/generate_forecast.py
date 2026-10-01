import pandas as pd
import numpy as np
import joblib

from pathlib import Path


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

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "restaurant_demand_model.joblib"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
)

OUTPUT_FILE = (
    REPORT_DIR
    / "future_demand_forecast.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "future_demand_forecast_summary.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_DAYS = 14


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def load_data():

    print(
        "\n[1/7] Loading historical forecasting data..."
    )

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["Date"]
    )

    df = df.sort_values(
        [
            "Outlet",
            "Date"
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Historical rows : {len(df):,}"
    )

    print(
        f"Last historical date : "
        f"{df['Date'].max().date()}"
    )

    print(
        f"Outlets : "
        f"{df['Outlet'].nunique()}"
    )

    return df


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\n[2/7] Loading trained demand model..."
    )

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"Model loaded:"
    )

    print(
        "HistGradientBoosting pipeline"
    )

    return model


# ============================================================
# CREATE FUTURE CALENDAR
# ============================================================

def create_future_calendar(
    df
):

    print(
        "\n[3/7] Creating future dates..."
    )

    last_date = (
        df["Date"].max()
    )

    future_dates = pd.date_range(
        start=last_date
        + pd.Timedelta(days=1),

        periods=FORECAST_DAYS,

        freq="D"
    )

    outlets = (
        sorted(
            df["Outlet"]
            .unique()
        )
    )

    future_rows = []

    for outlet in outlets:

        for date in future_dates:

            future_rows.append(
                {
                    "Date": date,
                    "Outlet": outlet
                }
            )

    future = pd.DataFrame(
        future_rows
    )

    print(
        f"Forecast dates : "
        f"{future_dates.min().date()} "
        f"to "
        f"{future_dates.max().date()}"
    )

    print(
        f"Future rows : "
        f"{len(future):,}"
    )

    return future


# ============================================================
# CALENDAR FEATURES
# ============================================================

def add_calendar_features(
    future
):

    future = future.copy()

    future["Year"] = (
        future["Date"].dt.year
    )

    future["Month"] = (
        future["Date"].dt.month
    )

    future["Day"] = (
        future["Date"].dt.day
    )

    future["DayOfWeek"] = (
        future["Date"].dt.dayofweek
    )

    future["WeekOfYear"] = (
        future["Date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    future["WeekOfMonth"] = (
        (future["Day"] - 1)
        // 7
        + 1
    )

    future["Quarter"] = (
        future["Date"].dt.quarter
    )

    future["IsWeekend"] = (
        future["DayOfWeek"] >= 5
    ).astype(int)

    return future


# ============================================================
# BUILD FORECAST FEATURES
# ============================================================

def create_features(
    historical,
    future_row,
    outlet_history
):

    date = future_row["Date"]

    outlet = future_row["Outlet"]

    # --------------------------------------------------------
    # History is maintained separately for each outlet.
    # --------------------------------------------------------

    history = (
        outlet_history[
            outlet
        ]
        .sort_values("Date")
        .copy()
    )

    orders = (
        history["Orders"]
        .tolist()
    )

    # --------------------------------------------------------
    # Need enough historical observations.
    # --------------------------------------------------------

    if len(orders) < 28:

        raise ValueError(
            f"Not enough historical observations "
            f"for outlet {outlet}"
        )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    lag_1 = orders[-1]

    lag_7 = orders[-7]

    lag_14 = orders[-14]

    lag_28 = orders[-28]

    # --------------------------------------------------------
    # Rolling features
    #
    # IMPORTANT:
    # Only historical observations are used.
    # The current forecast is NOT included.
    # --------------------------------------------------------

    rolling_7 = np.mean(
        orders[-7:]
    )

    rolling_14 = np.mean(
        orders[-14:]
    )

    rolling_28 = np.mean(
        orders[-28:]
    )

    row = {

        "Outlet": outlet,

        "DayOfWeek":
            date.dayofweek,

        "Month":
            date.month,

        "Day":
            date.day,

        "WeekOfYear":
            int(
                date.isocalendar().week
            ),

        "WeekOfMonth":
            ((date.day - 1) // 7) + 1,

        "Quarter":
            date.quarter,

        "IsWeekend":
            int(
                date.dayofweek >= 5
            ),

        "Orders_Lag_1":
            lag_1,

        "Orders_Lag_7":
            lag_7,

        "Orders_Lag_14":
            lag_14,

        "Orders_Lag_28":
            lag_28,

        "Orders_RollingMean_7":
            rolling_7,

        "Orders_RollingMean_14":
            rolling_14,

        "Orders_RollingMean_28":
            rolling_28
    }

    return row


# ============================================================
# RECURSIVE FORECASTING
# ============================================================

def generate_forecasts(
    historical,
    future,
    model
):

    print(
        "\n[4/7] Generating recursive forecasts..."
    )

    # --------------------------------------------------------
    # Keep independent history for each outlet.
    # --------------------------------------------------------

    outlet_history = {}

    for outlet in (
        historical["Outlet"]
        .unique()
    ):

        outlet_history[
            outlet
        ] = (
            historical[
                historical["Outlet"]
                == outlet
            ][
                [
                    "Date",
                    "Orders"
                ]
            ]
            .sort_values("Date")
            .copy()
        )

    forecast_rows = []

    # --------------------------------------------------------
    # Forecast date by date.
    # --------------------------------------------------------

    for date in sorted(
        future["Date"].unique()
    ):

        print(
            f"Forecasting "
            f"{pd.Timestamp(date).date()}..."
        )

        daily_rows = (
            future[
                future["Date"]
                == date
            ]
        )

        for _, future_row in (
            daily_rows.iterrows()
        ):

            outlet = (
                future_row["Outlet"]
            )

            feature_row = (
                create_features(
                    historical,
                    future_row,
                    outlet_history
                )
            )

            X_future = pd.DataFrame(
                [feature_row]
            )

            prediction = (
                model
                .predict(
                    X_future
                )[0]
            )

            # Demand cannot be negative.
            prediction = max(
                0,
                prediction
            )

            # Orders are discrete.
            predicted_orders = int(
                round(
                    prediction
                )
            )

            forecast_rows.append(
                {
                    "Date":
                        future_row["Date"],

                    "Outlet":
                        outlet,

                    "ForecastOrders":
                        predicted_orders
                }
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # The prediction becomes part of the history.
            # This allows the next future day to use it as
            # a lag/rolling feature.
            # ------------------------------------------------

            new_history_row = pd.DataFrame(
                [
                    {
                        "Date":
                            future_row["Date"],

                        "Orders":
                            predicted_orders
                    }
                ]
            )

            outlet_history[
                outlet
            ] = pd.concat(
                [
                    outlet_history[
                        outlet
                    ],
                    new_history_row
                ],
                ignore_index=True
            )

    forecast = pd.DataFrame(
        forecast_rows
    )

    return forecast


# ============================================================
# ADD BUSINESS FEATURES
# ============================================================

def add_business_features(
    forecast
):

    print(
        "\n[5/7] Adding business interpretation fields..."
    )

    forecast = forecast.copy()

    forecast["DayName"] = (
        forecast["Date"]
        .dt.day_name()
    )

    forecast["IsWeekend"] = (
        forecast["Date"]
        .dt.dayofweek >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Demand level
    #
    # Calculated relative to each outlet's forecast average.
    # --------------------------------------------------------

    outlet_mean = (
        forecast
        .groupby("Outlet")[
            "ForecastOrders"
        ]
        .transform("mean")
    )

    forecast["DemandIndex"] = np.where(
        outlet_mean > 0,

        forecast["ForecastOrders"]
        / outlet_mean
        * 100,

        0
    )

    forecast["DemandLevel"] = np.select(

        [
            forecast["DemandIndex"] >= 115,

            forecast["DemandIndex"] <= 85
        ],

        [
            "High",

            "Low"
        ],

        default="Normal"
    )

    return forecast


# ============================================================
# SAVE FORECAST
# ============================================================

def save_forecast(
    forecast
):

    print(
        "\n[6/7] Saving future forecast..."
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    forecast = forecast.sort_values(
        [
            "Date",
            "Outlet"
        ]
    ).reset_index(
        drop=True
    )

    forecast.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nForecast saved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        f"Rows : "
        f"{len(forecast):,}"
    )


# ============================================================
# CREATE SUMMARY
# ============================================================

def create_summary(
    forecast
):

    print(
        "\n[7/7] Creating forecast summary..."
    )

    summary = (
        forecast
        .groupby("Outlet")
        .agg(
            TotalForecastOrders=(
                "ForecastOrders",
                "sum"
            ),

            AverageDailyOrders=(
                "ForecastOrders",
                "mean"
            ),

            MaximumDailyOrders=(
                "ForecastOrders",
                "max"
            ),

            MinimumDailyOrders=(
                "ForecastOrders",
                "min"
            )
        )
        .reset_index()
    )

    high_demand_days = (
        forecast[
            forecast["DemandLevel"]
            == "High"
        ]
        .groupby("Outlet")
        .size()
        .reset_index(
            name="HighDemandDays"
        )
    )

    summary = summary.merge(
        high_demand_days,
        on="Outlet",
        how="left"
    )

    summary["HighDemandDays"] = (
        summary["HighDemandDays"]
        .fillna(0)
        .astype(int)
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "RESTAURANT FUTURE DEMAND FORECAST\n"
        )

        file.write(
            "========================================\n\n"
        )

        file.write(
            f"Forecast period: "
            f"{forecast['Date'].min().date()} "
            f"to "
            f"{forecast['Date'].max().date()}\n\n"
        )

        file.write(
            "Outlet-level forecast summary:\n\n"
        )

        file.write(
            summary.to_string(
                index=False
            )
        )

        file.write(
            "\n\n"
        )

        file.write(
            "Daily forecast:\n\n"
        )

        daily_summary = (
            forecast
            .groupby("Date")[
                "ForecastOrders"
            ]
            .sum()
            .reset_index()
        )

        file.write(
            daily_summary.to_string(
                index=False
            )
        )

    print(
        "\nSummary saved:"
    )

    print(
        SUMMARY_FILE
    )

    print(
        "\nOutlet summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "RESTAURANT SALES INTELLIGENCE"
    )

    print(
        "FUTURE DEMAND FORECASTING"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # 1
    # --------------------------------------------------------

    historical = load_data()

    # --------------------------------------------------------
    # 2
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # 3
    # --------------------------------------------------------

    future = create_future_calendar(
        historical
    )

    # --------------------------------------------------------
    # Add calendar features
    # --------------------------------------------------------

    future = add_calendar_features(
        future
    )

    # --------------------------------------------------------
    # 4
    # --------------------------------------------------------

    forecast = generate_forecasts(
        historical,
        future,
        model
    )

    # --------------------------------------------------------
    # 5
    # --------------------------------------------------------

    forecast = add_business_features(
        forecast
    )

    # --------------------------------------------------------
    # 6
    # --------------------------------------------------------

    save_forecast(
        forecast
    )

    # --------------------------------------------------------
    # 7
    # --------------------------------------------------------

    create_summary(
        forecast
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FUTURE FORECASTING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nGenerated files:"
    )

    print(
        "  - reports\\future_demand_forecast.csv"
    )

    print(
        "  - reports\\future_demand_forecast_summary.txt"
    )

    print(
        "\nForecast preview:"
    )

    print(
        forecast.to_string(
            index=False
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()