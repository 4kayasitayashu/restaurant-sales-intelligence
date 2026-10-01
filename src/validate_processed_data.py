import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "sales_processed.csv"


def main():
    print("=" * 80)
    print("RESTAURANT SALES INTELLIGENCE")
    print("PROCESSED DATA VALIDATION")
    print("=" * 80)

    if not PROCESSED_FILE.exists():
        print("\nERROR: Processed dataset not found:")
        print(PROCESSED_FILE)
        return

    df = pd.read_csv(PROCESSED_FILE)

    print(f"\nRows       : {len(df):,}")
    print(f"Columns    : {len(df.columns)}")

    print("\n" + "-" * 80)
    print("DATE RANGE")
    print("-" * 80)

    df["DateTime"] = pd.to_datetime(df["DateTime"])

    print(f"Start : {df['DateTime'].min()}")
    print(f"End   : {df['DateTime'].max()}")

    print("\n" + "-" * 80)
    print("OUTLETS")
    print("-" * 80)

    print(df["Outlet"].value_counts().to_string())

    print("\n" + "-" * 80)
    print("PAYMENT METHODS")
    print("-" * 80)

    print(df["PaymentMethod"].value_counts().to_string())

    print("\n" + "-" * 80)
    print("TIME PERIODS")
    print("-" * 80)

    print(df["TimePeriod"].value_counts().to_string())

    print("\n" + "-" * 80)
    print("KEY BUSINESS METRICS")
    print("-" * 80)

    total_revenue = df["Revenue"].sum()
    total_net_revenue = df["NetRevenue"].sum()
    total_orders = df["ReceiptNumber"].nunique()
    total_items = df["TotalItem"].sum()
    average_order_value = total_revenue / total_orders

    print(f"Total Revenue       : {total_revenue:,.2f}")
    print(f"Net Revenue         : {total_net_revenue:,.2f}")
    print(f"Total Orders        : {total_orders:,}")
    print(f"Total Items         : {total_items:,}")
    print(f"Average Order Value : {average_order_value:,.2f}")

    print("\n" + "-" * 80)
    print("LOYALTY USAGE")
    print("-" * 80)

    print(
        df["LoyaltyCustomer"]
        .value_counts()
        .to_string()
    )

    print("\n" + "-" * 80)
    print("DATA QUALITY CHECK")
    print("-" * 80)

    print(f"Missing values : {df.isnull().sum().sum():,}")
    print(f"Duplicate rows : {df.duplicated().sum():,}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()