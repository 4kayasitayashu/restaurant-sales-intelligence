import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

INPUT_FILE = PROCESSED_DIR / "item_sales_processed.csv"


# ============================================================
# OUTPUT FILES
# ============================================================

PRODUCT_REPORT = REPORT_DIR / "product_performance.csv"
CATEGORY_REPORT = REPORT_DIR / "category_performance.csv"
OUTLET_CATEGORY_REPORT = REPORT_DIR / "outlet_category_performance.csv"
MONTHLY_CATEGORY_REPORT = REPORT_DIR / "monthly_category_performance.csv"
PRODUCT_OUTLET_REPORT = REPORT_DIR / "product_outlet_performance.csv"
PRODUCT_MONTHLY_REPORT = REPORT_DIR / "product_monthly_performance.csv"
BUSINESS_INSIGHTS = REPORT_DIR / "product_business_insights.txt"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n[1/8] Loading item-level dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print("\n[2/8] Preparing analytical data...")

    df = df.copy()

    df["DateTime"] = pd.to_datetime(
        df["DateTime"],
        errors="coerce"
    )

    numeric_columns = [
        "Price",
        "ItemQuantity",
        "CatalogItemValue",
        "AllocatedRevenue",
        "AllocatedNetRevenue",
        "ReceiptTotalAmount",
        "ReceiptNetSales",
        "Month",
        "Year",
        "Hour"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df["ItemQuantity"] = (
        df["ItemQuantity"]
        .fillna(1)
    )

    return df


# ============================================================
# PRODUCT PERFORMANCE
# ============================================================

def create_product_report(df):

    print("\n[3/8] Creating product performance report...")

    product_report = (
        df
        .groupby(
            [
                "ProductId",
                "ProductName",
                "Variant",
                "Category"
            ],
            as_index=False
        )
        .agg(
            UnitsSold=(
                "ItemQuantity",
                "sum"
            ),

            CatalogRevenue=(
                "CatalogItemValue",
                "sum"
            ),

            AllocatedRevenue=(
                "AllocatedRevenue",
                "sum"
            ),

            AllocatedNetRevenue=(
                "AllocatedNetRevenue",
                "sum"
            ),

            AveragePrice=(
                "Price",
                "mean"
            ),

            Transactions=(
                "ReceiptNumber",
                "nunique"
            ),

            Outlets=(
                "Outlet",
                "nunique"
            )
        )
    )

    total_units = product_report[
        "UnitsSold"
    ].sum()

    total_revenue = product_report[
        "AllocatedRevenue"
    ].sum()

    if total_units > 0:

        product_report[
            "UnitSharePercent"
        ] = (
            product_report["UnitsSold"]
            / total_units
            * 100
        )

    else:

        product_report[
            "UnitSharePercent"
        ] = 0

    if total_revenue > 0:

        product_report[
            "RevenueSharePercent"
        ] = (
            product_report[
                "AllocatedRevenue"
            ]
            / total_revenue
            * 100
        )

    else:

        product_report[
            "RevenueSharePercent"
        ] = 0

    product_report = (
        product_report
        .sort_values(
            "UnitsSold",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return product_report


# ============================================================
# CATEGORY PERFORMANCE
# ============================================================

def create_category_report(df):

    print("\n[4/8] Creating category performance report...")

    category_report = (
        df
        .groupby(
            "Category",
            as_index=False
        )
        .agg(
            UnitsSold=(
                "ItemQuantity",
                "sum"
            ),

            CatalogRevenue=(
                "CatalogItemValue",
                "sum"
            ),

            AllocatedRevenue=(
                "AllocatedRevenue",
                "sum"
            ),

            AllocatedNetRevenue=(
                "AllocatedNetRevenue",
                "sum"
            ),

            Transactions=(
                "ReceiptNumber",
                "nunique"
            ),

            UniqueProducts=(
                "ProductId",
                "nunique"
            ),

            Outlets=(
                "Outlet",
                "nunique"
            ),

            AverageItemPrice=(
                "Price",
                "mean"
            )
        )
    )

    total_units = category_report[
        "UnitsSold"
    ].sum()

    total_revenue = category_report[
        "AllocatedRevenue"
    ].sum()

    category_report[
        "UnitSharePercent"
    ] = (
        category_report["UnitsSold"]
        / total_units
        * 100
    )

    category_report[
        "RevenueSharePercent"
    ] = (
        category_report["AllocatedRevenue"]
        / total_revenue
        * 100
    )

    category_report = (
        category_report
        .sort_values(
            "AllocatedRevenue",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return category_report


# ============================================================
# OUTLET × CATEGORY
# ============================================================

def create_outlet_category_report(df):

    print(
        "\n[5/8] Creating outlet × category analysis..."
    )

    report = (
        df
        .groupby(
            [
                "Outlet",
                "Category"
            ],
            as_index=False
        )
        .agg(
            UnitsSold=(
                "ItemQuantity",
                "sum"
            ),

            AllocatedRevenue=(
                "AllocatedRevenue",
                "sum"
            ),

            Transactions=(
                "ReceiptNumber",
                "nunique"
            ),

            UniqueProducts=(
                "ProductId",
                "nunique"
            )
        )
    )

    report[
        "RevenueShareWithinOutlet"
    ] = (
        report
        .groupby("Outlet")[
            "AllocatedRevenue"
        ]
        .transform(
            lambda x:
            x / x.sum() * 100
        )
    )

    return report.sort_values(
        [
            "Outlet",
            "AllocatedRevenue"
        ],
        ascending=[True, False]
    )


# ============================================================
# MONTHLY CATEGORY
# ============================================================

def create_monthly_category_report(df):

    print(
        "\n[6/8] Creating monthly category trends..."
    )

    report = (
        df
        .groupby(
            [
                "Year",
                "Month",
                "MonthName",
                "Category"
            ],
            as_index=False
        )
        .agg(
            UnitsSold=(
                "ItemQuantity",
                "sum"
            ),

            AllocatedRevenue=(
                "AllocatedRevenue",
                "sum"
            ),

            Transactions=(
                "ReceiptNumber",
                "nunique"
            )
        )
    )

    report = report.sort_values(
        [
            "Year",
            "Month",
            "AllocatedRevenue"
        ],
        ascending=[True, True, False]
    )

    return report


# ============================================================
# PRODUCT × OUTLET
# ============================================================

def create_product_outlet_report(df):

    print(
        "\n[7/8] Creating product × outlet analysis..."
    )

    report = (
        df
        .groupby(
            [
                "Outlet",
                "ProductId",
                "ProductName",
                "Variant",
                "Category"
            ],
            as_index=False
        )
        .agg(
            UnitsSold=(
                "ItemQuantity",
                "sum"
            ),

            AllocatedRevenue=(
                "AllocatedRevenue",
                "sum"
            ),

            Transactions=(
                "ReceiptNumber",
                "nunique"
            )
        )
    )

    return report.sort_values(
        "AllocatedRevenue",
        ascending=False
    )


# ============================================================
# PRODUCT × MONTH
# ============================================================

def create_product_monthly_report(df):

    print(
        "\nCreating product × monthly trends..."
    )

    report = (
        df
        .groupby(
            [
                "Year",
                "Month",
                "MonthName",
                "ProductId",
                "ProductName",
                "Variant",
                "Category"
            ],
            as_index=False
        )
        .agg(
            UnitsSold=(
                "ItemQuantity",
                "sum"
            ),

            AllocatedRevenue=(
                "AllocatedRevenue",
                "sum"
            )
        )
    )

    return report.sort_values(
        [
            "Year",
            "Month",
            "AllocatedRevenue"
        ],
        ascending=[True, True, False]
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

def generate_business_insights(
    df,
    product_report,
    category_report
):

    print(
        "\nGenerating business insights..."
    )

    lines = []

    lines.append(
        "RESTAURANT SALES INTELLIGENCE"
    )

    lines.append(
        "PRODUCT & CATEGORY ANALYSIS"
    )

    lines.append("=" * 70)

    lines.append(
        "\nDATASET SUMMARY"
    )

    lines.append(
        f"Item records: {len(df):,}"
    )

    lines.append(
        f"Unique products sold: "
        f"{df['ProductId'].nunique():,}"
    )

    lines.append(
        f"Categories: "
        f"{df['Category'].nunique():,}"
    )

    lines.append(
        f"Outlets: "
        f"{df['Outlet'].nunique():,}"
    )

    # --------------------------------------------------------
    # TOP PRODUCTS
    # --------------------------------------------------------

    lines.append(
        "\nTOP PRODUCTS BY UNITS SOLD"
    )

    top_products = (
        product_report
        .head(10)
    )

    for _, row in top_products.iterrows():

        lines.append(
            f"- {row['ProductName']} "
            f"({row['Variant']}): "
            f"{row['UnitsSold']:,.0f} units"
        )

    # --------------------------------------------------------
    # TOP PRODUCTS BY REVENUE
    # --------------------------------------------------------

    lines.append(
        "\nTOP PRODUCTS BY ALLOCATED REVENUE"
    )

    top_revenue_products = (
        product_report
        .sort_values(
            "AllocatedRevenue",
            ascending=False
        )
        .head(10)
    )

    for _, row in top_revenue_products.iterrows():

        lines.append(
            f"- {row['ProductName']} "
            f"({row['Variant']}): "
            f"{row['AllocatedRevenue']:,.2f}"
        )

    # --------------------------------------------------------
    # CATEGORY SUMMARY
    # --------------------------------------------------------

    lines.append(
        "\nCATEGORY PERFORMANCE"
    )

    for _, row in category_report.iterrows():

        lines.append(
            f"- {row['Category']}: "
            f"{row['UnitsSold']:,.0f} units, "
            f"{row['AllocatedRevenue']:,.2f} "
            f"allocated revenue, "
            f"{row['RevenueSharePercent']:.2f}% "
            f"revenue share"
        )

    # --------------------------------------------------------
    # PRODUCT CONCENTRATION
    # --------------------------------------------------------

    total_revenue = (
        product_report[
            "AllocatedRevenue"
        ].sum()
    )

    top_10_revenue = (
        product_report
        .sort_values(
            "AllocatedRevenue",
            ascending=False
        )
        .head(10)[
            "AllocatedRevenue"
        ]
        .sum()
    )

    if total_revenue > 0:

        top_10_share = (
            top_10_revenue
            / total_revenue
            * 100
        )

        lines.append(
            "\nPRODUCT CONCENTRATION"
        )

        lines.append(
            f"Top 10 products account for "
            f"{top_10_share:.2f}% of "
            f"allocated revenue."
        )

    # --------------------------------------------------------
    # PRICE VS VOLUME
    # --------------------------------------------------------

    if len(product_report) > 1:

        correlation = (
            product_report[
                [
                    "AveragePrice",
                    "UnitsSold"
                ]
            ]
            .corr()
            .loc[
                "AveragePrice",
                "UnitsSold"
            ]
        )

        lines.append(
            "\nPRICE VS UNIT DEMAND"
        )

        lines.append(
            f"Correlation between average "
            f"catalog price and units sold: "
            f"{correlation:.4f}"
        )

        lines.append(
            "This correlation describes the "
            "observed dataset relationship; "
            "it does not establish causation."
        )

    # --------------------------------------------------------
    # METHODOLOGY NOTE
    # --------------------------------------------------------

    lines.append(
        "\nREVENUE METHODOLOGY"
    )

    lines.append(
        "Transaction revenue is available at "
        "receipt level."
    )

    lines.append(
        "Product-level revenue is therefore "
        "allocated using each item's catalog "
        "price share within the receipt."
    )

    lines.append(
        "The item pipeline achieved 100% "
        "revenue coverage against transaction "
        "totals."
    )

    lines.append(
        "Allocated product revenue should be "
        "interpreted as an analytical allocation, "
        "not a separately recorded POS line-item "
        "revenue field."
    )

    return "\n".join(lines)


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    product_report,
    category_report,
    outlet_category_report,
    monthly_category_report,
    product_outlet_report,
    product_monthly_report,
    insights
):

    print(
        "\n[8/8] Saving analytical reports..."
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    product_report.to_csv(
        PRODUCT_REPORT,
        index=False
    )

    category_report.to_csv(
        CATEGORY_REPORT,
        index=False
    )

    outlet_category_report.to_csv(
        OUTLET_CATEGORY_REPORT,
        index=False
    )

    monthly_category_report.to_csv(
        MONTHLY_CATEGORY_REPORT,
        index=False
    )

    product_outlet_report.to_csv(
        PRODUCT_OUTLET_REPORT,
        index=False
    )

    product_monthly_report.to_csv(
        PRODUCT_MONTHLY_REPORT,
        index=False
    )

    with open(
        BUSINESS_INSIGHTS,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(insights)

    print("\nReports created:")

    print(PRODUCT_REPORT)
    print(CATEGORY_REPORT)
    print(OUTLET_CATEGORY_REPORT)
    print(MONTHLY_CATEGORY_REPORT)
    print(PRODUCT_OUTLET_REPORT)
    print(PRODUCT_MONTHLY_REPORT)
    print(BUSINESS_INSIGHTS)


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary(
    product_report,
    category_report
):

    print(
        "\n" + "=" * 80
    )

    print(
        "PRODUCT & CATEGORY ANALYSIS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nProducts analysed : "
        f"{len(product_report):,}"
    )

    print(
        f"Categories        : "
        f"{len(category_report):,}"
    )

    print(
        "\nTop 10 products by units:"
    )

    display_columns = [
        "ProductName",
        "Variant",
        "Category",
        "UnitsSold",
        "AllocatedRevenue"
    ]

    print(
        product_report[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nCategory performance:"
    )

    category_columns = [
        "Category",
        "UnitsSold",
        "AllocatedRevenue",
        "RevenueSharePercent"
    ]

    print(
        category_report[
            category_columns
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("RESTAURANT SALES INTELLIGENCE")
    print("PRODUCT & CATEGORY ANALYTICS")
    print("=" * 80)

    # 1
    df = load_data()

    # 2
    df = prepare_data(df)

    # 3
    product_report = create_product_report(
        df
    )

    # 4
    category_report = create_category_report(
        df
    )

    # 5
    outlet_category_report = (
        create_outlet_category_report(df)
    )

    # 6
    monthly_category_report = (
        create_monthly_category_report(df)
    )

    # 7
    product_outlet_report = (
        create_product_outlet_report(df)
    )

    product_monthly_report = (
        create_product_monthly_report(df)
    )

    # Business insights
    insights = generate_business_insights(
        df,
        product_report,
        category_report
    )

    # Save
    save_reports(
        product_report,
        category_report,
        outlet_category_report,
        monthly_category_report,
        product_outlet_report,
        product_monthly_report,
        insights
    )

    # Summary
    print_summary(
        product_report,
        category_report
    )


if __name__ == "__main__":
    main()