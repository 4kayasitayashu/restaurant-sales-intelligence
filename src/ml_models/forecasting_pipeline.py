import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "sales_processed.csv"

OUTPUT_FILE = (
    PROCESSED_DIR
    / "daily_demand_forecasting.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n[1/7] Loading processed sales data...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

    return df


# ============================================================
# PREPARE DATE
# ============================================================

def prepare_date(df):

    print(
        "\n[2/7] Preparing date information..."
    )

    df = df.copy()

    df["DateTime"] = pd.to_datetime(
        df["DateTime"],
        errors="coerce"
    )

    df["Date"] = (
        df["DateTime"]
        .dt.date
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    return df


# ============================================================
# DAILY AGGREGATION
# ============================================================

def create_daily_dataset(df):

    print(
        "\n[3/7] Creating daily demand dataset..."
    )

    daily = (
        df
        .groupby(
            [
                "Date",
                "Outlet"
            ],
            as_index=False
        )
        .agg(
            Orders=(
                "ReceiptNumber",
                "nunique"
            ),

            ItemsSold=(
                "TotalItem",
                "sum"
            ),

            Revenue=(
                "TotalAmount",
                "sum"
            ),

            NetRevenue=(
                "NetSales",
                "sum"
            ),

            Tax=(
                "Tax",
                "sum"
            )
        )
    )

    daily["AOV"] = np.where(
        daily["Orders"] > 0,
        daily["Revenue"]
        / daily["Orders"],
        0
    )

    return daily


# ============================================================
# CALENDAR FEATURES
# ============================================================

def create_calendar_features(daily):

    print(
        "\n[4/7] Creating calendar features..."
    )

    daily = daily.copy()

    daily["Year"] = (
        daily["Date"].dt.year
    )

    daily["Month"] = (
        daily["Date"].dt.month
    )

    daily["MonthName"] = (
        daily["Date"]
        .dt.month_name()
    )

    daily["Day"] = (
        daily["Date"].dt.day
    )

    daily["DayOfWeek"] = (
        daily["Date"].dt.dayofweek
    )

    daily["DayName"] = (
        daily["Date"]
        .dt.day_name()
    )

    daily["WeekOfYear"] = (
        daily["Date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    daily["WeekOfMonth"] = (
        (daily["Day"] - 1)
        // 7
        + 1
    )

    daily["IsWeekend"] = (
        daily["DayOfWeek"] >= 5
    ).astype(int)

    daily["Quarter"] = (
        daily["Date"].dt.quarter
    )

    return daily


# ============================================================
# TIME-SERIES COMPLETENESS
# ============================================================

def check_date_continuity(daily):

    print(
        "\n[5/7] Checking date continuity..."
    )

    all_dates = pd.date_range(
        daily["Date"].min(),
        daily["Date"].max(),
        freq="D"
    )

    actual_dates = pd.DatetimeIndex(
        daily["Date"].unique()
    )

    missing_dates = (
        all_dates
        .difference(actual_dates)
    )

    print(
        f"Date range : "
        f"{daily['Date'].min().date()} "
        f"to "
        f"{daily['Date'].max().date()}"
    )

    print(
        f"Calendar days : "
        f"{len(all_dates):,}"
    )

    print(
        f"Observed dates : "
        f"{len(actual_dates):,}"
    )

    print(
        f"Missing dates : "
        f"{len(missing_dates):,}"
    )

    if len(missing_dates) > 0:

        print(
            "\nMissing dates:"
        )

        for date in missing_dates[:20]:

            print(
                date.date()
            )

    return missing_dates


# ============================================================
# LAG FEATURES
# ============================================================

def create_lag_features(daily):

    print(
        "\n[6/7] Creating leakage-safe lag features..."
    )

    daily = daily.copy()

    daily = daily.sort_values(
        [
            "Outlet",
            "Date"
        ]
    )

    # --------------------------------------------------------
    # Previous-day demand
    # --------------------------------------------------------

    daily["Orders_Lag_1"] = (
        daily
        .groupby("Outlet")["Orders"]
        .shift(1)
    )

    daily["Revenue_Lag_1"] = (
        daily
        .groupby("Outlet")["Revenue"]
        .shift(1)
    )

    daily["Items_Lag_1"] = (
        daily
        .groupby("Outlet")["ItemsSold"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Previous 7 days
    # --------------------------------------------------------

    daily["Orders_Lag_7"] = (
        daily
        .groupby("Outlet")["Orders"]
        .shift(7)
    )

    daily["Revenue_Lag_7"] = (
        daily
        .groupby("Outlet")["Revenue"]
        .shift(7)
    )

    daily["Items_Lag_7"] = (
        daily
        .groupby("Outlet")["ItemsSold"]
        .shift(7)
    )

    # --------------------------------------------------------
    # Previous 14 days
    # --------------------------------------------------------

    daily["Orders_Lag_14"] = (
        daily
        .groupby("Outlet")["Orders"]
        .shift(14)
    )

    daily["Revenue_Lag_14"] = (
        daily
        .groupby("Outlet")["Revenue"]
        .shift(14)
    )

    # --------------------------------------------------------
    # Previous 28 days
    # --------------------------------------------------------

    daily["Orders_Lag_28"] = (
        daily
        .groupby("Outlet")["Orders"]
        .shift(28)
    )

    daily["Revenue_Lag_28"] = (
        daily
        .groupby("Outlet")["Revenue"]
        .shift(28)
    )

    # --------------------------------------------------------
    # Rolling demand features
    #
    # IMPORTANT:
    # shift(1) happens BEFORE rolling calculation.
    #
    # Therefore today's target is never included
    # in today's rolling features.
    # --------------------------------------------------------

    previous_orders = (
        daily
        .groupby("Outlet")["Orders"]
        .shift(1)
    )

    previous_revenue = (
        daily
        .groupby("Outlet")["Revenue"]
        .shift(1)
    )

    daily["Orders_RollingMean_7"] = (
        previous_orders
        .groupby(daily["Outlet"])
        .transform(
            lambda x:
            x.rolling(
                window=7,
                min_periods=1
            ).mean()
        )
    )

    daily["Orders_RollingMean_14"] = (
        previous_orders
        .groupby(daily["Outlet"])
        .transform(
            lambda x:
            x.rolling(
                window=14,
                min_periods=1
            ).mean()
        )
    )

    daily["Orders_RollingMean_28"] = (
        previous_orders
        .groupby(daily["Outlet"])
        .transform(
            lambda x:
            x.rolling(
                window=28,
                min_periods=1
            ).mean()
        )
    )

    daily["Revenue_RollingMean_7"] = (
        previous_revenue
        .groupby(daily["Outlet"])
        .transform(
            lambda x:
            x.rolling(
                window=7,
                min_periods=1
            ).mean()
        )
    )

    daily["Revenue_RollingMean_28"] = (
        previous_revenue
        .groupby(daily["Outlet"])
        .transform(
            lambda x:
            x.rolling(
                window=28,
                min_periods=1
            ).mean()
        )
    )

    return daily


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(daily):

    print(
        "\n[7/7] Validating forecasting dataset..."
    )

    daily = daily.copy()

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    daily = daily.sort_values(
        [
            "Outlet",
            "Date"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicates = (
        daily
        .duplicated(
            subset=[
                "Date",
                "Outlet"
            ]
        )
        .sum()
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = (
        daily
        .isna()
        .sum()
        .sum()
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        f"Daily outlet rows : "
        f"{len(daily):,}"
    )

    print(
        f"Duplicate Date+Outlet rows : "
        f"{duplicates:,}"
    )

    print(
        f"Total missing values : "
        f"{missing:,}"
    )

    print(
        "\nRows by outlet:"
    )

    print(
        daily["Outlet"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nDate range by outlet:"
    )

    date_summary = (
        daily
        .groupby("Outlet")["Date"]
        .agg(
            StartDate="min",
            EndDate="max",
            Days="nunique"
        )
    )

    print(
        date_summary.to_string()
    )

    return daily


# ============================================================
# SAVE
# ============================================================

def save_dataset(daily):

    print(
        "\nSaving forecasting dataset..."
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    daily.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        f"Rows    : {len(daily):,}"
    )

    print(
        f"Columns : {len(daily.columns):,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("RESTAURANT SALES INTELLIGENCE")
    print("FORECASTING DATASET PIPELINE")
    print("=" * 80)

    # 1
    df = load_data()

    # 2
    df = prepare_date(df)

    # 3
    daily = create_daily_dataset(
        df
    )

    # 4
    daily = create_calendar_features(
        daily
    )

    # 5
    check_date_continuity(
        daily
    )

    # 6
    daily = create_lag_features(
        daily
    )

    # 7
    daily = validate_dataset(
        daily
    )

    save_dataset(
        daily
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FORECASTING DATASET PIPELINE COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()