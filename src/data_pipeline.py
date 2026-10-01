import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TRANSACTIONS_FILE = RAW_DIR / "Transactions.csv"
PRODUCTS_FILE = RAW_DIR / "Products.csv"

OUTPUT_FILE = PROCESSED_DIR / "sales_processed.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("\n[1/6] Loading datasets...")

    transactions = pd.read_csv(TRANSACTIONS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)

    print(f"Transactions : {len(transactions):,}")
    print(f"Products     : {len(products):,}")

    return transactions, products


# ============================================================
# CLEAN TRANSACTION DATA
# ============================================================

def clean_transactions(df):
    print("\n[2/6] Cleaning transaction data...")

    df = df.copy()

    df.columns = df.columns.str.strip()

    text_columns = [
        "Outlet",
        "Date",
        "Time",
        "ReceiptNumber",
        "Items",
        "PaymentMethod"
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # Combine Date + Time
    df["DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        errors="coerce"
    )

    invalid_datetime = df["DateTime"].isna().sum()

    if invalid_datetime > 0:
        print(
            f"Invalid datetime rows removed: "
            f"{invalid_datetime}"
        )

        df = df.dropna(
            subset=["DateTime"]
        )

    # Numeric columns
    numeric_columns = [
        "NetSales",
        "Tax",
        "TotalAmount",
        "TotalItem",
        "UseLoyaltyCard"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove invalid critical records
    df = df.dropna(
        subset=[
            "NetSales",
            "TotalAmount",
            "TotalItem"
        ]
    )

    # Valid item quantity
    df = df[
        df["TotalItem"] > 0
    ]

    # Valid revenue
    df = df[
        df["TotalAmount"] >= 0
    ]

    print(
        f"Rows after cleaning: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# CREATE TIME FEATURES
# ============================================================

def create_time_features(df):
    print("\n[3/6] Creating time features...")

    df = df.copy()

    df["Year"] = (
        df["DateTime"].dt.year
    )

    df["Month"] = (
        df["DateTime"].dt.month
    )

    df["MonthName"] = (
        df["DateTime"].dt.month_name()
    )

    df["Day"] = (
        df["DateTime"].dt.day
    )

    df["DayOfWeek"] = (
        df["DateTime"].dt.dayofweek
    )

    df["DayName"] = (
        df["DateTime"].dt.day_name()
    )

    df["WeekOfYear"] = (
        df["DateTime"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["WeekOfMonth"] = (
        (df["Day"] - 1) // 7
    ) + 1

    df["IsWeekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)

    df["Hour"] = (
        df["DateTime"].dt.hour
    )

    df["Minute"] = (
        df["DateTime"].dt.minute
    )

    def get_time_period(hour):

        if hour < 7:
            return "Early Morning"

        if hour < 12:
            return "Morning"

        if hour < 15:
            return "Lunch"

        if hour < 18:
            return "Afternoon"

        if hour < 21:
            return "Evening"

        return "Night"

    df["TimePeriod"] = (
        df["Hour"].apply(
            get_time_period
        )
    )

    df["Quarter"] = (
        df["DateTime"].dt.quarter
    )

    return df


# ============================================================
# CREATE BUSINESS METRICS
# ============================================================

def create_business_metrics(df):
    print("\n[4/6] Creating business metrics...")

    df = df.copy()

    # Transaction revenue including tax
    df["Revenue"] = (
        df["TotalAmount"]
    )

    # Revenue before tax
    df["NetRevenue"] = (
        df["NetSales"]
    )

    # Tax amount
    df["TaxAmount"] = (
        df["Tax"]
    )

    # Revenue generated per item
    df["RevenuePerItem"] = (
        df["Revenue"]
        / df["TotalItem"].replace(
            0,
            np.nan
        )
    )

    # Loyalty indicator
    df["LoyaltyCustomer"] = np.where(
        df["UseLoyaltyCard"] == 1,
        "Yes",
        "No"
    )

    return df


# ============================================================
# PREPARE PRODUCT CATALOG
# ============================================================

def prepare_products(products):
    print("\n[5/6] Preparing product catalog...")

    products = products.copy()

    products.columns = (
        products.columns
        .str.strip()
    )

    text_columns = [
        "ProductName",
        "Variant",
        "Category",
        "Description"
    ]

    for column in text_columns:
        products[column] = (
            products[column]
            .astype(str)
            .str.strip()
        )

    products["Price"] = pd.to_numeric(
        products["Price"],
        errors="coerce"
    )

    return products


# ============================================================
# CREATE ITEM SUMMARY
# ============================================================

def create_item_summary(df):
    df = df.copy()

    # Count product names appearing inside each
    # transaction's Items field.
    #
    # Example:
    #
    # "Coffee, Sandwich"
    #
    # becomes a count of 2.
    #
    # We intentionally do NOT store the Python list
    # inside the final DataFrame because CSV storage
    # of list objects is not useful for analytics.

    df["ParsedItemCount"] = (
        df["Items"]
        .fillna("")
        .astype(str)
        .apply(
            lambda x: len(
                [
                    item.strip()
                    for item in x.split(",")
                    if item.strip()
                ]
            )
        )
    )

    return df


# ============================================================
# FINAL DATA QUALITY CHECK
# ============================================================

def validate_processed_data(df):

    print("\nData quality check:")

    missing_values = (
        df.isnull()
        .sum()
        .sum()
    )

    print(
        f"Missing values : "
        f"{missing_values:,}"
    )

    # Only hashable/scalar columns are used for
    # duplicate detection.
    #
    # The pipeline currently contains only scalar
    # columns, but this is intentionally written
    # defensively.

    hashable_columns = []

    for column in df.columns:

        if not df[column].apply(
            lambda value: isinstance(
                value,
                (list, dict, set)
            )
        ).any():

            hashable_columns.append(
                column
            )

    duplicate_count = (
        df[hashable_columns]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate rows : "
        f"{duplicate_count:,}"
    )


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(df):

    print("\n[6/6] Saving processed dataset...")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nProcessed file saved:"
    )

    print(OUTPUT_FILE)

    print(
        f"\nFinal rows    : "
        f"{len(df):,}"
    )

    print(
        f"Final columns : "
        f"{len(df.columns)}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 80)
    print("RESTAURANT SALES INTELLIGENCE")
    print("DATA CLEANING & FEATURE ENGINEERING PIPELINE")
    print("=" * 80)

    # 1. Load
    transactions, products = (
        load_data()
    )

    # 2. Clean
    transactions = (
        clean_transactions(
            transactions
        )
    )

    # 3. Time features
    transactions = (
        create_time_features(
            transactions
        )
    )

    # 4. Business metrics
    transactions = (
        create_business_metrics(
            transactions
        )
    )

    # 5. Product catalog preparation
    products = (
        prepare_products(
            products
        )
    )

    # 6. Transaction item summary
    transactions = (
        create_item_summary(
            transactions
        )
    )

    # Final validation
    validate_processed_data(
        transactions
    )

    # Save
    save_processed_data(
        transactions
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    print("\nProcessed features include:")

    features = [
        "DateTime",
        "Year",
        "Month",
        "MonthName",
        "Quarter",
        "Day",
        "DayOfWeek",
        "DayName",
        "WeekOfYear",
        "WeekOfMonth",
        "IsWeekend",
        "Hour",
        "Minute",
        "TimePeriod",
        "Revenue",
        "NetRevenue",
        "TaxAmount",
        "RevenuePerItem",
        "LoyaltyCustomer",
        "ParsedItemCount"
    ]

    for feature in features:
        print(
            f"  - {feature}"
        )

    print(
        "\nRaw CSV files were NOT modified."
    )


if __name__ == "__main__":
    main()