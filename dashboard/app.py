
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Restaurant Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SALES_FILE = BASE_DIR / "data" / "processed" / "sales_processed.csv"
ITEM_FILE = BASE_DIR / "data" / "processed" / "item_sales_processed.csv"

PRODUCT_FILE = BASE_DIR / "reports" / "product_performance.csv"
CATEGORY_FILE = BASE_DIR / "reports" / "category_performance.csv"

FORECAST_FILE = BASE_DIR / "reports" / "future_demand_forecast.csv"
DAILY_INSIGHTS_FILE = BASE_DIR / "reports" / "daily_demand_insights.csv"

MODEL_FILE = BASE_DIR / "reports" / "forecast_model_comparison.csv"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #f7f8fa;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .dashboard-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .dashboard-subtitle {
            color: #68707d;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 650;
            margin-top: 1rem;
            margin-bottom: 0.8rem;
        }

        .metric-note {
            color: #6b7280;
            font-size: 0.85rem;
        }

        .footer {
            text-align: center;
            color: #7a828e;
            font-size: 0.8rem;
            padding-top: 2rem;
            padding-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def load_data():
    sales = pd.read_csv(SALES_FILE)
    items = pd.read_csv(ITEM_FILE)
    products = pd.read_csv(PRODUCT_FILE)
    categories = pd.read_csv(CATEGORY_FILE)
    forecast = pd.read_csv(FORECAST_FILE)
    daily_insights = pd.read_csv(DAILY_INSIGHTS_FILE)
    model_comparison = pd.read_csv(MODEL_FILE)

    sales["DateTime"] = pd.to_datetime(sales["DateTime"], errors="coerce")
    sales["Date"] = sales["DateTime"].dt.date

    forecast["Date"] = pd.to_datetime(
        forecast["Date"], errors="coerce"
    )

    daily_insights["Date"] = pd.to_datetime(
        daily_insights["Date"], errors="coerce"
    )

    return (
        sales,
        items,
        products,
        categories,
        forecast,
        daily_insights,
        model_comparison,
    )


try:
    (
        sales,
        items,
        products,
        categories,
        forecast,
        daily_insights,
        model_comparison,
    ) = load_data()

except Exception as e:
    st.error("Dashboard could not load the required data files.")
    st.exception(e)
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Restaurant Intelligence")

st.sidebar.caption(
    "Sales analytics, product intelligence and demand forecasting"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Sales Analytics",
        "Product Intelligence",
        "Outlet Intelligence",
        "Demand Forecast",
        "Business Insights",
    ],
)

st.sidebar.divider()

st.sidebar.caption("Dataset")
st.sidebar.write("Anonymous F&B POS Dataset")
st.sidebar.write("Jan 2025 – Sep 2025")

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">Restaurant Sales Intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dashboard-subtitle">'
    "Historical sales analytics + product intelligence + "
    "outlet analysis + machine-learning demand forecasting"
    "</div>",
    unsafe_allow_html=True,
)

# ============================================================
# COMMON METRICS
# ============================================================

total_revenue = sales["Revenue"].sum()
total_orders = len(sales)
total_items = sales["TotalItem"].sum()
aov = total_revenue / total_orders if total_orders else 0

forecast_total = forecast["ForecastOrders"].sum()

# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.markdown(
        '<div class="section-title">Business Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Revenue",
        f"Rp {total_revenue:,.0f}",
    )

    col2.metric(
        "Total Orders",
        f"{total_orders:,}",
    )

    col3.metric(
        "Average Order Value",
        f"Rp {aov:,.0f}",
    )

    col4.metric(
        "Total Items Sold",
        f"{total_items:,.0f}",
    )

    st.divider()

    # Monthly performance
    monthly = (
        sales.assign(
            Month=sales["DateTime"].dt.to_period("M").astype(str)
        )
        .groupby("Month")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("ReceiptNumber", "count"),
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Revenue")

        fig = px.line(
            monthly,
            x="Month",
            y="Revenue",
            markers=True,
            labels={
                "Revenue": "Revenue (IDR)",
                "Month": "Month",
            },
        )

        fig.update_layout(
            height=380,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monthly Orders")

        fig = px.bar(
            monthly,
            x="Month",
            y="Orders",
            labels={
                "Orders": "Orders",
                "Month": "Month",
            },
        )

        fig.update_layout(height=380)

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Outlet Performance")

    outlet_summary = (
        sales.groupby("Outlet")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("ReceiptNumber", "count"),
            Items=("TotalItem", "sum"),
        )
        .reset_index()
    )

    outlet_summary["AOV"] = (
        outlet_summary["Revenue"]
        / outlet_summary["Orders"]
    )

    outlet_summary["Revenue Share %"] = (
        outlet_summary["Revenue"]
        / outlet_summary["Revenue"].sum()
        * 100
    )

    outlet_display = outlet_summary.copy()

    outlet_display["Revenue"] = outlet_display["Revenue"].map(
        lambda x: f"Rp {x:,.0f}"
    )

    outlet_display["AOV"] = outlet_display["AOV"].map(
        lambda x: f"Rp {x:,.0f}"
    )

    outlet_display["Revenue Share %"] = outlet_display[
        "Revenue Share %"
    ].map(lambda x: f"{x:.2f}%")

    st.dataframe(
        outlet_display,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Forecast Snapshot")

    fc1, fc2, fc3 = st.columns(3)

    fc1.metric(
        "Next 14-Day Forecast Orders",
        f"{forecast_total:,.0f}",
    )

    fc2.metric(
        "Average Daily Forecast",
        f"{forecast_total / 14:,.1f}",
    )

    fc3.metric(
        "Forecasted Outlets",
        f"{forecast['Outlet'].nunique()}",
    )

# ============================================================
# SALES ANALYTICS
# ============================================================

elif page == "Sales Analytics":

    st.markdown(
        '<div class="section-title">Sales Analytics</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    # Day of week
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday = (
        sales.groupby("DayName")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("ReceiptNumber", "count"),
        )
        .reindex(weekday_order)
        .reset_index()
    )

    with col1:

        st.subheader("Revenue by Day of Week")

        fig = px.bar(
            weekday,
            x="DayName",
            y="Revenue",
            labels={
                "DayName": "Day",
                "Revenue": "Revenue (IDR)",
            },
        )

        fig.update_layout(height=380)

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Orders by Day of Week")

        fig = px.bar(
            weekday,
            x="DayName",
            y="Orders",
            labels={
                "DayName": "Day",
                "Orders": "Orders",
            },
        )

        fig.update_layout(height=380)

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hourly Demand")

    hourly = (
        sales.groupby("Hour")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("ReceiptNumber", "count"),
        )
        .reset_index()
    )

    fig = px.line(
        hourly,
        x="Hour",
        y="Orders",
        markers=True,
        labels={
            "Hour": "Hour of Day",
            "Orders": "Orders",
        },
    )

    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Payment Method")

        payment = (
            sales.groupby("PaymentMethod")
            .agg(
                Revenue=("Revenue", "sum"),
                Orders=("ReceiptNumber", "count"),
            )
            .reset_index()
        )

        fig = px.bar(
            payment,
            x="PaymentMethod",
            y="Revenue",
            labels={
                "PaymentMethod": "Payment Method",
                "Revenue": "Revenue (IDR)",
            },
        )

        fig.update_layout(height=360)

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Time Period Demand")

        period = (
            sales.groupby("TimePeriod")
            .agg(
                Revenue=("Revenue", "sum"),
                Orders=("ReceiptNumber", "count"),
            )
            .reset_index()
        )

        period_order = [
            "Early Morning",
            "Morning",
            "Lunch",
            "Afternoon",
            "Evening",
            "Night",
        ]

        period["TimePeriod"] = pd.Categorical(
            period["TimePeriod"],
            categories=period_order,
            ordered=True,
        )

        period = period.sort_values("TimePeriod")

        fig = px.bar(
            period,
            x="TimePeriod",
            y="Orders",
            labels={
                "TimePeriod": "Time Period",
                "Orders": "Orders",
            },
        )

        fig.update_layout(height=360)

        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PRODUCT INTELLIGENCE
# ============================================================

elif page == "Product Intelligence":

    st.markdown(
        '<div class="section-title">Product Intelligence</div>',
        unsafe_allow_html=True,
    )

    top_products = (
        products.sort_values(
            "UnitsSold",
            ascending=False,
        )
        .head(10)
        .copy()
    )

    st.subheader("Top 10 Products by Units Sold")

    fig = px.bar(
        top_products.sort_values("UnitsSold"),
        x="UnitsSold",
        y="ProductName",
        orientation="h",
        labels={
            "UnitsSold": "Units Sold",
            "ProductName": "Product",
        },
    )

    fig.update_layout(height=480)

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Category Revenue Share")

        fig = px.pie(
            categories,
            names="Category",
            values="AllocatedRevenue",
            hole=0.45,
        )

        fig.update_layout(height=420)

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Category Performance")

        category_display = categories.copy()

        category_display["AllocatedRevenue"] = (
            category_display["AllocatedRevenue"]
            .map(lambda x: f"Rp {x:,.0f}")
        )

        category_display["RevenueSharePercent"] = (
            category_display["RevenueSharePercent"]
            .map(lambda x: f"{x:.2f}%")
        )

        st.dataframe(
            category_display,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Product Performance")

    product_display = products.copy()

    if "AllocatedRevenue" in product_display.columns:
        product_display["AllocatedRevenue"] = (
            product_display["AllocatedRevenue"]
            .map(lambda x: f"Rp {x:,.0f}")
        )

    st.dataframe(
        product_display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Product revenue is analytically allocated from transaction-level "
        "revenue using catalog-price shares because the source POS data "
        "does not provide separate line-item revenue."
    )

# ============================================================
# OUTLET INTELLIGENCE
# ============================================================

elif page == "Outlet Intelligence":

    st.markdown(
        '<div class="section-title">Outlet Intelligence</div>',
        unsafe_allow_html=True,
    )

    outlet = (
        sales.groupby("Outlet")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("ReceiptNumber", "count"),
            Items=("TotalItem", "sum"),
        )
        .reset_index()
    )

    outlet["AOV"] = (
        outlet["Revenue"]
        / outlet["Orders"]
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Revenue by Outlet")

        fig = px.bar(
            outlet,
            x="Outlet",
            y="Revenue",
            labels={
                "Revenue": "Revenue (IDR)",
                "Outlet": "Outlet",
            },
        )

        fig.update_layout(height=380)

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Orders by Outlet")

        fig = px.bar(
            outlet,
            x="Outlet",
            y="Orders",
            labels={
                "Orders": "Orders",
                "Outlet": "Outlet",
            },
        )

        fig.update_layout(height=380)

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Outlet KPI Summary")

    outlet_display = outlet.copy()

    outlet_display["Revenue"] = outlet_display[
        "Revenue"
    ].map(lambda x: f"Rp {x:,.0f}")

    outlet_display["AOV"] = outlet_display[
        "AOV"
    ].map(lambda x: f"Rp {x:,.0f}")

    st.dataframe(
        outlet_display,
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# DEMAND FORECAST
# ============================================================

elif page == "Demand Forecast":

    st.markdown(
        '<div class="section-title">Machine Learning Demand Forecast</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Forecast horizon: 14 days after the last historical observation. "
        "Model: HistGradientBoosting."
    )

    outlet_options = ["All Outlets"] + sorted(
        forecast["Outlet"].dropna().unique().tolist()
    )

    selected_outlet = st.selectbox(
        "Select Outlet",
        outlet_options,
    )

    if selected_outlet == "All Outlets":

        selected_forecast = (
            forecast.groupby("Date")
            .agg(
                ForecastOrders=("ForecastOrders", "sum"),
            )
            .reset_index()
        )

        title = "Network Demand Forecast"

    else:

        selected_forecast = forecast[
            forecast["Outlet"] == selected_outlet
        ].copy()

        title = f"{selected_outlet} Demand Forecast"

    st.subheader(title)

    fig = px.line(
        selected_forecast,
        x="Date",
        y="ForecastOrders",
        markers=True,
        labels={
            "Date": "Date",
            "ForecastOrders": "Forecast Orders",
        },
    )

    fig.update_layout(
        height=430,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    if selected_outlet == "All Outlets":

        forecast_total_selected = (
            selected_forecast["ForecastOrders"].sum()
        )

        average_daily = (
            selected_forecast["ForecastOrders"].mean()
        )

        peak_daily = (
            selected_forecast["ForecastOrders"].max()
        )

        fc1, fc2, fc3 = st.columns(3)

        fc1.metric(
            "14-Day Forecast",
            f"{forecast_total_selected:,.0f} orders",
        )

        fc2.metric(
            "Average Daily Demand",
            f"{average_daily:,.1f}",
        )

        fc3.metric(
            "Peak Daily Demand",
            f"{peak_daily:,.0f}",
        )

    else:

        forecast_total_selected = (
            selected_forecast["ForecastOrders"].sum()
        )

        average_daily = (
            selected_forecast["ForecastOrders"].mean()
        )

        peak_daily = (
            selected_forecast["ForecastOrders"].max()
        )

        low_daily = (
            selected_forecast["ForecastOrders"].min()
        )

        high_days = (
            selected_forecast["DemandLevel"]
            .eq("High")
            .sum()
            if "DemandLevel" in selected_forecast.columns
            else 0
        )

        fc1, fc2, fc3, fc4 = st.columns(4)

        fc1.metric(
            "14-Day Forecast",
            f"{forecast_total_selected:,.0f}",
        )

        fc2.metric(
            "Average / Day",
            f"{average_daily:,.1f}",
        )

        fc3.metric(
            "Peak / Day",
            f"{peak_daily:,.0f}",
        )

        fc4.metric(
            "High-Demand Days",
            f"{high_days}",
        )

    st.subheader("Forecast Details")

    forecast_display = selected_forecast.copy()

    if "DemandLevel" in forecast_display.columns:
        forecast_display["DemandLevel"] = (
            forecast_display["DemandLevel"]
            .fillna("Normal")
        )

    st.dataframe(
        forecast_display,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Model Performance")

    model_display = model_comparison.copy()

    rename_map = {
        "Model": "Model",
        "MAE": "MAE",
        "RMSE": "RMSE",
        "WAPE_Percent": "WAPE (%)",
        "SMAPE_Percent": "sMAPE (%)",
    }

    model_display = model_display.rename(
        columns=rename_map
    )

    st.dataframe(
        model_display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Evaluation uses a chronological holdout period "
        "from 2025-08-08 to 2025-09-30. "
        "Lower MAE, RMSE, WAPE and sMAPE indicate lower forecast error."
    )

# ============================================================
# BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":

    st.markdown(
        '<div class="section-title">Business Decision Support</div>',
        unsafe_allow_html=True,
    )

    st.subheader("14-Day Forecast Summary")

    summary = (
        forecast.groupby("Outlet")
        .agg(
            ForecastOrders=("ForecastOrders", "sum"),
            AverageDailyOrders=("ForecastOrders", "mean"),
            PeakDailyOrders=("ForecastOrders", "max"),
            LowestDailyOrders=("ForecastOrders", "min"),
        )
        .reset_index()
    )

    summary["ForecastSharePercent"] = (
        summary["ForecastOrders"]
        / summary["ForecastOrders"].sum()
        * 100
    )

    summary_display = summary.copy()

    summary_display["ForecastSharePercent"] = (
        summary_display["ForecastSharePercent"]
        .map(lambda x: f"{x:.2f}%")
    )

    st.dataframe(
        summary_display,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("High-Demand Outlet Days")

    if "DemandLevel" in forecast.columns:

        high_demand = forecast[
            forecast["DemandLevel"] == "High"
        ].copy()

        if len(high_demand) > 0:

            st.dataframe(
                high_demand,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.success(
                "No high-demand outlet-days were flagged "
                "during the forecast horizon."
            )

    st.subheader("Operational Signals")

    for _, row in summary.iterrows():

        outlet_name = row["Outlet"]
        share = row["ForecastSharePercent"]
        avg_orders = row["AverageDailyOrders"]

        if share > 45:

            message = (
                f"{outlet_name}: Maintain strong operational capacity. "
                f"It represents approximately {share:.1f}% of forecast demand."
            )

        elif share < 20:

            message = (
                f"{outlet_name}: Monitor demand closely and avoid "
                f"unnecessary overstaffing. Forecast share is "
                f"approximately {share:.1f}%."
            )

        else:

            message = (
                f"{outlet_name}: Maintain normal operations. "
                f"Average forecast demand is {avg_orders:.1f} orders/day."
            )

        st.info(message)

    st.subheader("Top Forecast Days")

    daily = (
        forecast.groupby("Date")
        .agg(
            TotalForecastOrders=("ForecastOrders", "sum"),
        )
        .reset_index()
        .sort_values(
            "TotalForecastOrders",
            ascending=False,
        )
        .head(5)
    )

    st.dataframe(
        daily,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Management Recommendations")

    recommendations = [
        "Use the 14-day demand forecast for short-term staffing and operational planning.",
        "Maintain sufficient capacity at outlets contributing the largest share of forecast demand.",
        "Monitor lower-volume outlets carefully to reduce unnecessary staffing and inventory pressure.",
        "Use product-level demand signals to support inventory prioritisation.",
        "Re-run the forecasting pipeline when new actual sales data becomes available.",
        "Compare forecasted demand with actual demand after the forecast period to continuously monitor model performance.",
    ]

    for recommendation in recommendations:
        st.markdown(f"• {recommendation}")

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Restaurant Sales Intelligence & Demand Forecasting |
        Historical POS Analytics + Machine Learning |
        Forecasts are estimates and should be validated against actual future sales.
    </div>
    """,
    unsafe_allow_html=True,
)
