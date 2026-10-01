import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

PROCESSED_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "sales_processed.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n[1/8] Loading processed dataset...")

    df = pd.read_csv(
        PROCESSED_FILE
    )

    df["DateTime"] = pd.to_datetime(
        df["DateTime"]
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    return df


# ============================================================
# EXECUTIVE KPIs
# ============================================================

def executive_kpis(df):

    print("\n" + "=" * 80)
    print("EXECUTIVE BUSINESS KPIs")
    print("=" * 80)

    total_revenue = (
        df["Revenue"].sum()
    )

    net_revenue = (
        df["NetRevenue"].sum()
    )

    total_orders = (
        df["ReceiptNumber"].nunique()
    )

    total_items = (
        df["TotalItem"].sum()
    )

    aov = (
        total_revenue
        / total_orders
    )

    revenue_per_item = (
        total_revenue
        / total_items
    )

    print(
        f"\nTotal Revenue       : "
        f"{total_revenue:,.2f}"
    )

    print(
        f"Net Revenue         : "
        f"{net_revenue:,.2f}"
    )

    print(
        f"Total Orders        : "
        f"{total_orders:,}"
    )

    print(
        f"Total Items         : "
        f"{total_items:,}"
    )

    print(
        f"Average Order Value : "
        f"{aov:,.2f}"
    )

    print(
        f"Revenue / Item      : "
        f"{revenue_per_item:,.2f}"
    )


# ============================================================
# OUTLET ANALYSIS
# ============================================================

def outlet_analysis(df):

    print("\n" + "=" * 80)
    print("OUTLET PERFORMANCE")
    print("=" * 80)

    outlet = (
        df.groupby("Outlet")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("ReceiptNumber", "nunique"),
            Items=("TotalItem", "sum")
        )
        .reset_index()
    )

    outlet["AOV"] = (
        outlet["Revenue"]
        / outlet["Orders"]
    )

    outlet["RevenueShare"] = (
        outlet["Revenue"]
        / outlet["Revenue"].sum()
        * 100
    )

    outlet = outlet.sort_values(
        "Revenue",
        ascending=False
    )

    print(
        "\n"
        + outlet.to_string(
            index=False,
            formatters={
                "Revenue": "{:,.2f}".format,
                "AOV": "{:,.2f}".format,
                "RevenueShare": "{:.2f}%".format
            }
        )
    )


# ============================================================
# MONTHLY ANALYSIS
# ============================================================

def monthly_analysis(df):

    print("\n" + "=" * 80)
    print("MONTHLY PERFORMANCE")
    print("=" * 80)

    monthly = (
        df.groupby(
            ["Year", "Month", "MonthName"]
        )
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("ReceiptNumber", "nunique"),
            Items=("TotalItem", "sum")
        )
        .reset_index()
    )

    monthly["AOV"] = (
        monthly["Revenue"]
        / monthly["Orders"]
    )

    monthly["RevenueGrowth"] = (
        monthly["Revenue"]
        .pct_change()
        * 100
    )

    print(
        "\n"
        + monthly.to_string(
            index=False,
            formatters={
                "Revenue": "{:,.2f}".format,
                "AOV": "{:,.2f}".format,
                "RevenueGrowth": "{:.2f}%".format
            }
        )
    )


# ============================================================
# DAY OF WEEK ANALYSIS
# ============================================================

def day_analysis(df):

    print("\n" + "=" * 80)
    print("DAY OF WEEK PERFORMANCE")
    print("=" * 80)

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    daily = (
        df.groupby("DayName")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("ReceiptNumber", "nunique"),
            Items=("TotalItem", "sum")
        )
        .reindex(day_order)
        .reset_index()
    )

    daily["AOV"] = (
        daily["Revenue"]
        / daily["Orders"]
    )

    print(
        "\n"
        + daily.to_string(
            index=False,
            formatters={
                "Revenue": "{:,.2f}".format,
                "AOV": "{:,.2f}".format
            }
        )
    )


# ============================================================
# HOURLY DEMAND ANALYSIS
# ============================================================

def hourly_analysis(df):

    print("\n" + "=" * 80)
    print("HOURLY DEMAND")
    print("=" * 80)

    hourly = (
        df.groupby("Hour")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("ReceiptNumber", "nunique"),
            Items=("TotalItem", "sum")
        )
        .reset_index()
    )

    hourly["AOV"] = (
        hourly["Revenue"]
        / hourly["Orders"]
    )

    hourly = hourly.sort_values(
        "Orders",
        ascending=False
    )

    print(
        "\n"
        + hourly.to_string(
            index=False,
            formatters={
                "Revenue": "{:,.2f}".format,
                "AOV": "{:,.2f}".format
            }
        )
    )


# ============================================================
# TIME PERIOD ANALYSIS
# ============================================================

def time_period_analysis(df):

    print("\n" + "=" * 80)
    print("TIME PERIOD PERFORMANCE")
    print("=" * 80)

    period_order = [
        "Early Morning",
        "Morning",
        "Lunch",
        "Afternoon",
        "Evening",
        "Night"
    ]

    period = (
        df.groupby("TimePeriod")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("ReceiptNumber", "nunique"),
            Items=("TotalItem", "sum")
        )
        .reindex(period_order)
        .reset_index()
    )

    period["AOV"] = (
        period["Revenue"]
        / period["Orders"]
    )

    print(
        "\n"
        + period.to_string(
            index=False,
            formatters={
                "Revenue": "{:,.2f}".format,
                "AOV": "{:,.2f}".format
            }
        )
    )


# ============================================================
# PAYMENT ANALYSIS
# ============================================================

def payment_analysis(df):

    print("\n" + "=" * 80)
    print("PAYMENT METHOD ANALYSIS")
    print("=" * 80)

    payment = (
        df.groupby("PaymentMethod")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("ReceiptNumber", "nunique")
        )
        .reset_index()
    )

    payment["RevenueShare"] = (
        payment["Revenue"]
        / payment["Revenue"].sum()
        * 100
    )

    payment["OrderShare"] = (
        payment["Orders"]
        / payment["Orders"].sum()
        * 100
    )

    payment = payment.sort_values(
        "Revenue",
        ascending=False
    )

    print(
        "\n"
        + payment.to_string(
            index=False,
            formatters={
                "Revenue": "{:,.2f}".format,
                "RevenueShare": "{:.2f}%".format,
                "OrderShare": "{:.2f}%".format
            }
        )
    )


# ============================================================
# LOYALTY ANALYSIS
# ============================================================

def loyalty_analysis(df):

    print("\n" + "=" * 80)
    print("LOYALTY CARD ANALYSIS")
    print("=" * 80)

    loyalty = (
        df.groupby("LoyaltyCustomer")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("ReceiptNumber", "nunique"),
            Items=("TotalItem", "sum")
        )
        .reset_index()
    )

    loyalty["AOV"] = (
        loyalty["Revenue"]
        / loyalty["Orders"]
    )

    loyalty["RevenueShare"] = (
        loyalty["Revenue"]
        / loyalty["Revenue"].sum()
        * 100
    )

    print(
        "\n"
        + loyalty.to_string(
            index=False,
            formatters={
                "Revenue": "{:,.2f}".format,
                "AOV": "{:,.2f}".format,
                "RevenueShare": "{:.2f}%".format
            }
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("RESTAURANT SALES INTELLIGENCE")
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 80)

    df = load_data()

    executive_kpis(df)

    outlet_analysis(df)

    monthly_analysis(df)

    day_analysis(df)

    hourly_analysis(df)

    time_period_analysis(df)

    payment_analysis(df)

    loyalty_analysis(df)

    print("\n" + "=" * 80)
    print("EDA REPORT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()