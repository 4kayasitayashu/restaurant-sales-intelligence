import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

FORECAST_FILE = (
    BASE_DIR
    / "reports"
    / "future_demand_forecast.csv"
)

PRODUCT_FILE = (
    BASE_DIR
    / "reports"
    / "product_performance.csv"
)

CATEGORY_FILE = (
    BASE_DIR
    / "reports"
    / "category_performance.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "reports"
    / "business_insights.csv"
)

SUMMARY_FILE = (
    BASE_DIR
    / "reports"
    / "business_recommendations.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n[1/6] Loading analytics data...")

    forecast = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["Date"]
    )

    products = pd.read_csv(
        PRODUCT_FILE
    )

    categories = pd.read_csv(
        CATEGORY_FILE
    )

    print(
        f"Forecast rows : {len(forecast):,}"
    )

    print(
        f"Products : {len(products):,}"
    )

    print(
        f"Categories : {len(categories):,}"
    )

    return forecast, products, categories


# ============================================================
# OUTLET INSIGHTS
# ============================================================

def outlet_insights(forecast):

    print("\n[2/6] Creating outlet insights...")

    outlet_summary = (
        forecast
        .groupby("Outlet")
        .agg(
            ForecastOrders=(
                "ForecastOrders",
                "sum"
            ),

            AverageDailyOrders=(
                "ForecastOrders",
                "mean"
            ),

            PeakDailyOrders=(
                "ForecastOrders",
                "max"
            ),

            LowestDailyOrders=(
                "ForecastOrders",
                "min"
            ),

            HighDemandDays=(
                "DemandLevel",
                lambda x: (
                    x == "High"
                ).sum()
            ),

            LowDemandDays=(
                "DemandLevel",
                lambda x: (
                    x == "Low"
                ).sum()
            )
        )
        .reset_index()
    )

    total_forecast = (
        outlet_summary[
            "ForecastOrders"
        ].sum()
    )

    outlet_summary[
        "ForecastSharePercent"
    ] = np.where(
        total_forecast > 0,

        outlet_summary[
            "ForecastOrders"
        ]
        / total_forecast
        * 100,

        0
    )

    # --------------------------------------------------------
    # Operational signal
    # --------------------------------------------------------

    outlet_summary[
        "OperationalSignal"
    ] = np.select(

        [
            outlet_summary[
                "HighDemandDays"
            ] >= 3,

            outlet_summary[
                "AverageDailyOrders"
            ] >= 75,

            outlet_summary[
                "AverageDailyOrders"
            ] <= 40
        ],

        [
            "Prepare for elevated demand",

            "Maintain strong operational capacity",

            "Monitor demand and avoid overstaffing"
        ],

        default="Maintain normal operations"
    )

    return outlet_summary


# ============================================================
# DAILY DEMAND INSIGHTS
# ============================================================

def daily_insights(forecast):

    print("\n[3/6] Creating daily demand insights...")

    daily = (
        forecast
        .groupby("Date")
        .agg(
            TotalForecastOrders=(
                "ForecastOrders",
                "sum"
            ),

            OutletsWithHighDemand=(
                "DemandLevel",
                lambda x: (
                    x == "High"
                ).sum()
            ),

            OutletsWithLowDemand=(
                "DemandLevel",
                lambda x: (
                    x == "Low"
                ).sum()
            )
        )
        .reset_index()
    )

    daily[
        "DemandStatus"
    ] = np.select(

        [
            daily[
                "OutletsWithHighDemand"
            ] >= 2,

            daily[
                "OutletsWithLowDemand"
            ] >= 2
        ],

        [
            "Multi-outlet high demand",

            "Multi-outlet low demand"
        ],

        default="Normal demand"
    )

    return daily


# ============================================================
# PRODUCT INSIGHTS
# ============================================================

def product_insights(products):

    print("\n[4/6] Creating product insights...")

    products = products.copy()

    products = products.sort_values(
        "UnitsSold",
        ascending=False
    )

    top_products = (
        products
        .head(10)
        .copy()
    )

    top_products[
        "BusinessSignal"
    ] = "High-volume product"

    return top_products


# ============================================================
# CATEGORY INSIGHTS
# ============================================================

def category_insights(categories):

    categories = categories.copy()

    categories = categories.sort_values(
        "AllocatedRevenue",
        ascending=False
    )

    total_revenue = (
        categories[
            "AllocatedRevenue"
        ].sum()
    )

    categories[
        "RevenueSharePercent"
    ] = np.where(
        total_revenue > 0,

        categories[
            "AllocatedRevenue"
        ]
        / total_revenue
        * 100,

        0
    )

    categories[
        "BusinessSignal"
    ] = np.select(

        [
            categories[
                "RevenueSharePercent"
            ] >= 25,

            categories[
                "RevenueSharePercent"
            ] <= 5
        ],

        [
            "Major revenue contributor",

            "Smaller revenue contributor"
        ],

        default="Important revenue contributor"
    )

    return categories


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    forecast,
    outlet_summary,
    daily,
    top_products,
    categories
):

    print("\n[5/6] Generating business recommendations...")

    recommendations = []

    # --------------------------------------------------------
    # Overall forecast
    # --------------------------------------------------------

    total_orders = int(
        forecast[
            "ForecastOrders"
        ].sum()
    )

    average_daily_orders = (
        forecast[
            "ForecastOrders"
        ]
        .sum()
        / forecast[
            "Date"
        ].nunique()
    )

    recommendations.append(
        "FORECAST OVERVIEW"
    )

    recommendations.append(
        f"The 14-day forecast indicates approximately "
        f"{total_orders:,} total orders across all outlets."
    )

    recommendations.append(
        f"Expected average demand is approximately "
        f"{average_daily_orders:.1f} orders per day across the network."
    )

    # --------------------------------------------------------
    # Outlet recommendations
    # --------------------------------------------------------

    recommendations.append(
        "\nOUTLET OPERATIONS"
    )

    for _, row in outlet_summary.iterrows():

        recommendations.append(
            f"{row['Outlet']}: "
            f"approximately "
            f"{row['AverageDailyOrders']:.1f} orders/day, "
            f"with a forecast range of "
            f"{int(row['LowestDailyOrders'])}–"
            f"{int(row['PeakDailyOrders'])} orders/day. "
            f"Signal: {row['OperationalSignal']}."
        )

    # --------------------------------------------------------
    # Peak demand dates
    # --------------------------------------------------------

    peak_days = (
        daily
        .sort_values(
            "TotalForecastOrders",
            ascending=False
        )
        .head(5)
    )

    recommendations.append(
        "\nPEAK DEMAND DATES"
    )

    for _, row in peak_days.iterrows():

        recommendations.append(
            f"{row['Date'].date()}: "
            f"{int(row['TotalForecastOrders'])} "
            f"forecast orders — "
            f"{row['DemandStatus']}."
        )

    # --------------------------------------------------------
    # Low demand dates
    # --------------------------------------------------------

    low_days = (
        daily
        .sort_values(
            "TotalForecastOrders",
            ascending=True
        )
        .head(5)
    )

    recommendations.append(
        "\nLOWER DEMAND DATES"
    )

    for _, row in low_days.iterrows():

        recommendations.append(
            f"{row['Date'].date()}: "
            f"{int(row['TotalForecastOrders'])} "
            f"forecast orders."
        )

    # --------------------------------------------------------
    # Product recommendations
    # --------------------------------------------------------

    recommendations.append(
        "\nPRODUCT SIGNALS"
    )

    for _, row in top_products.head(5).iterrows():

        recommendations.append(
            f"{row['ProductName']} "
            f"({row['Category']}): "
            f"{int(row['UnitsSold'])} units sold historically."
        )

    # --------------------------------------------------------
    # Category recommendations
    # --------------------------------------------------------

    recommendations.append(
        "\nCATEGORY SIGNALS"
    )

    for _, row in categories.iterrows():

        recommendations.append(
            f"{row['Category']}: "
            f"{row['RevenueSharePercent']:.2f}% "
            f"of allocated category revenue — "
            f"{row['BusinessSignal']}."
        )

    # --------------------------------------------------------
    # Operational guidance
    # --------------------------------------------------------

    recommendations.append(
        "\nOPERATIONAL GUIDANCE"
    )

    recommendations.append(
        "Use higher-demand forecast days to review staffing, "
        "ingredient preparation and operational capacity."
    )

    recommendations.append(
        "Use lower-demand periods to avoid unnecessary "
        "over-preparation and excessive staffing."
    )

    recommendations.append(
        "High-volume products should receive additional "
        "attention during preparation and inventory planning."
    )

    recommendations.append(
        "Forecast values are model-generated estimates and "
        "should be reviewed against actual demand as new "
        "sales data becomes available."
    )

    return recommendations


# ============================================================
# SAVE BUSINESS INSIGHTS
# ============================================================

def save_outputs(
    outlet_summary,
    daily,
    top_products,
    categories,
    recommendations
):

    print("\n[6/6] Saving business intelligence outputs...")

    # --------------------------------------------------------
    # Combined outlet insight file
    # --------------------------------------------------------

    outlet_output = outlet_summary.copy()

    outlet_output[
        "InsightType"
    ] = "Outlet Forecast"

    outlet_output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Detailed recommendation report
    # --------------------------------------------------------

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "RESTAURANT BUSINESS INTELLIGENCE REPORT\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            "Generated from historical sales analytics "
            "and 14-day ML demand forecast.\n\n"
        )

        for line in recommendations:

            file.write(
                line
                + "\n"
            )

        file.write(
            "\n"
            + "=" * 70
            + "\n"
        )

        file.write(
            "END OF REPORT\n"
        )

    # --------------------------------------------------------
    # Additional detailed CSVs
    # --------------------------------------------------------

    daily.to_csv(
        BASE_DIR
        / "reports"
        / "daily_demand_insights.csv",
        index=False
    )

    top_products.to_csv(
        BASE_DIR
        / "reports"
        / "top_product_insights.csv",
        index=False
    )

    categories.to_csv(
        BASE_DIR
        / "reports"
        / "category_business_insights.csv",
        index=False
    )

    print(
        "\nGenerated:"
    )

    print(
        "  - reports\\business_insights.csv"
    )

    print(
        "  - reports\\business_recommendations.txt"
    )

    print(
        "  - reports\\daily_demand_insights.csv"
    )

    print(
        "  - reports\\top_product_insights.csv"
    )

    print(
        "  - reports\\category_business_insights.csv"
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
        "BUSINESS DECISION LAYER"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # 1
    # --------------------------------------------------------

    forecast, products, categories = (
        load_data()
    )

    # --------------------------------------------------------
    # 2
    # --------------------------------------------------------

    outlet_summary = outlet_insights(
        forecast
    )

    # --------------------------------------------------------
    # 3
    # --------------------------------------------------------

    daily = daily_insights(
        forecast
    )

    # --------------------------------------------------------
    # 4
    # --------------------------------------------------------

    top_products = product_insights(
        products
    )

    categories = category_insights(
        categories
    )

    # --------------------------------------------------------
    # 5
    # --------------------------------------------------------

    recommendations = (
        generate_recommendations(
            forecast,
            outlet_summary,
            daily,
            top_products,
            categories
        )
    )

    # --------------------------------------------------------
    # 6
    # --------------------------------------------------------

    save_outputs(
        outlet_summary,
        daily,
        top_products,
        categories,
        recommendations
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BUSINESS DECISION LAYER COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nOutlet Summary:"
    )

    print(
        outlet_summary.to_string(
            index=False
        )
    )

    print(
        "\nTop Product Signals:"
    )

    print(
        top_products[
            [
                "ProductName",
                "Category",
                "UnitsSold"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nTop Forecast Days:"
    )

    print(
        daily
        .sort_values(
            "TotalForecastOrders",
            ascending=False
        )
        .head(5)
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )


if __name__ == "__main__":

    main()