import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import FuncFormatter

# ----------------------------
# Helpers
# ----------------------------
def _find_date_column(df, default="Date"):
    if default in df.columns:
        return default
    candidates = [c for c in df.columns if "date" in c.lower()]
    if not candidates:
        raise ValueError(f"No date-like column found. Columns are: {list(df.columns)}")
    return candidates[0]

def parse_dates_no_warning(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    dt = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    missing = dt.isna()
    if missing.any():
        dt2 = pd.to_datetime(s[missing], format="%d/%m/%y", errors="coerce")
        dt.loc[missing] = dt2
    return dt

def format_compact(value):
    try:
        v = float(value)
    except Exception:
        return str(value)
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f}k"
    return f"{v:.0f}"

def safe_margin_pct(net_profit, revenue):
    denom = revenue.where(revenue != 0)
    return (net_profit / denom) * 100

# ----------------------------
# Core chart function
# ----------------------------
def create_visualization(
    Sales_base: pd.DataFrame,
    Cost_dated: pd.DataFrame,
    cost_method="Moving Average",
    view_type="Month",
    metric_type="Amount",
):
    # ----- Sales -----
    Sales_base = Sales_base.copy()
    sales_date_col = _find_date_column(Sales_base, default="Date")

    Sales_base["Date"] = parse_dates_no_warning(Sales_base[sales_date_col])
    Sales_base["YearMonth"] = Sales_base["Date"].dt.to_period("M")

    Sales_base["Qty"] = pd.to_numeric(Sales_base["Qty"], errors="coerce").fillna(0.0)
    Sales_base["UnitPrice"] = pd.to_numeric(Sales_base["UnitPrice"], errors="coerce")
    Sales_base["Revenue"] = Sales_base["Qty"] * Sales_base["UnitPrice"]

    # ----- Cost -----
    Cost_dated = Cost_dated.copy()
    cost_date_col = _find_date_column(Cost_dated, default="Date")

    Cost_dated["Date"] = parse_dates_no_warning(Cost_dated[cost_date_col])
    Cost_dated["YearMonth"] = Cost_dated["Date"].dt.to_period("M")
    Cost_dated["CostPerUnit"] = pd.to_numeric(Cost_dated["CostPerUnit"], errors="coerce")

    # ----- Full month range over SALES -----
    sales_months = Sales_base["YearMonth"].dropna()
    if sales_months.empty:
        raise ValueError("Sales data has no valid dates to build a month range.")
    full_month_range = pd.period_range(sales_months.min(), sales_months.max(), freq="M")
    months_ts = full_month_range.to_timestamp()  # DatetimeIndex

    # ----- Monthly cost + cumulative moving average (count ALL months) -----
    monthly_cost = (
        Cost_dated.groupby("YearMonth", as_index=False)["CostPerUnit"]
        .mean()
        .set_index("YearMonth")
    )

    monthly_cost = monthly_cost.reindex(full_month_range)
    monthly_cost.index.name = "YearMonth"  # IMPORTANT: keep name after reindex

    cost_values = monthly_cost["CostPerUnit"].astype("float64").fillna(0.0)
    cum_sum = cost_values.cumsum()
    month_number = pd.Series(range(1, len(cost_values) + 1), index=monthly_cost.index, dtype="float64")
    monthly_cost["MovingAvgCost"] = (cum_sum / month_number).astype("float64")

    tmp = monthly_cost.reset_index()
    # Safety: if pandas calls it 'index' instead of 'YearMonth'
    if "YearMonth" not in tmp.columns and "index" in tmp.columns:
        tmp = tmp.rename(columns={"index": "YearMonth"})
    moving_avg_cost_df = tmp[["YearMonth", "MovingAvgCost"]]

    # ----- Apply cost method (needed for ProfitMargin%) -----
    if cost_method == "Total Average":
        total_avg_cost = Cost_dated["CostPerUnit"].mean()
        Sales_temp = Sales_base.copy()
        Sales_temp["TotalCost"] = Sales_temp["Qty"] * total_avg_cost
        Sales_temp["NetProfit"] = Sales_temp["Revenue"] - Sales_temp["TotalCost"]
    else:
        Sales_temp = Sales_base.merge(moving_avg_cost_df, on="YearMonth", how="left")
        Sales_temp["TotalCost"] = Sales_temp["Qty"] * Sales_temp["MovingAvgCost"]
        Sales_temp["NetProfit"] = Sales_temp["Revenue"] - Sales_temp["TotalCost"]

    # ----- Styling (kept from your code) -----
    region_order = ["North", "East", "West", "South"]
    revenue_colors = {"North": "#00CED1", "East": "#DC143C", "West": "#1E90FF", "South": "#FF8C00"}
    cost_colors    = {"North": "#A9A9A9", "East": "#808080", "West": "#696969", "South": "#C0C0C0"}

    # =========================================================
    # COUNT MODE
    # =========================================================
    if metric_type == "Count":
        if view_type == "Region":
            region_qty = Sales_temp.groupby("Region", dropna=False)["Qty"].sum().reset_index()
            region_qty["Region"] = pd.Categorical(region_qty["Region"], categories=region_order, ordered=True)
            region_qty = region_qty.sort_values("Region")

            region_amt = (
                Sales_temp.groupby("Region", dropna=False)
                .agg({"Revenue": "sum", "NetProfit": "sum"})
                .reset_index()
            )
            region_amt["ProfitMargin"] = safe_margin_pct(region_amt["NetProfit"], region_amt["Revenue"])
            region_amt["Region"] = pd.Categorical(region_amt["Region"], categories=region_order, ordered=True)
            region_amt = region_amt.sort_values("Region")

            fig, ax1 = plt.subplots(figsize=(18, 8))
            x_pos = list(range(len(region_order)))

            ax1.bar(
                x_pos,
                region_qty["Qty"].fillna(0.0),
                alpha=0.85,
                color=[revenue_colors[r] for r in region_order],
                label="Units (Qty)",
            )

            for i, val in enumerate(region_qty["Qty"].fillna(0.0).values):
                if val > 0:
                    ax1.text(i, val, format_compact(val), ha="center", va="bottom",
                             fontsize=10, fontweight="bold")

            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(region_order)
            ax1.set_xlabel("Region", fontsize=12, fontweight="bold")
            ax1.set_ylabel("Units Sold (Qty)", fontsize=12, fontweight="bold")
            ax1.grid(True, axis="y", alpha=0.3)
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))

            ax2 = ax1.twinx()
            ax2.plot(
                x_pos,
                region_amt["ProfitMargin"],
                marker="o",
                label="Profit Margin (%)",
                color="#2E8B57",
                linewidth=3,
                markersize=8,
            )
            ax2.set_ylabel("Profit Margin (%)", fontsize=12, fontweight="bold")
            ax2.set_ylim(-100, 100)
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, ncol=2)

            plt.title(
                f"Units Sold and Profit Margin by Region (Cost Method: {cost_method})",
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()
            return fig

        # Month view: Qty bars by region, ProfitMargin% line overall
        region_month_qty = Sales_temp.groupby(["Region", "YearMonth"], dropna=False)["Qty"].sum().reset_index()
        region_month_qty["YearMonth"] = region_month_qty["YearMonth"].dt.to_timestamp()

        monthly_amt = (
            Sales_temp.groupby("YearMonth", dropna=False)
            .agg({"Revenue": "sum", "NetProfit": "sum"})
            .reset_index()
        )
        monthly_amt["YearMonth"] = monthly_amt["YearMonth"].dt.to_timestamp()
        monthly_amt = monthly_amt.set_index("YearMonth").reindex(months_ts, fill_value=0).reset_index()
        monthly_amt.rename(columns={"index": "YearMonth"}, inplace=True)
        monthly_amt["ProfitMargin"] = safe_margin_pct(monthly_amt["NetProfit"], monthly_amt["Revenue"])

        bar_width = 0.2
        x_pos = list(range(len(months_ts)))

        all_combinations = pd.MultiIndex.from_product([region_order, months_ts], names=["Region", "YearMonth"])
        complete_qty = (
            region_month_qty.set_index(["Region", "YearMonth"])
            .reindex(all_combinations, fill_value=0)
            .reset_index()
        )

        fig, ax1 = plt.subplots(figsize=(18, 8))

        for i, region in enumerate(region_order):
            rd = complete_qty[complete_qty["Region"] == region].sort_values("YearMonth")
            x_offset = [x + (i * bar_width) for x in x_pos]

            bars = ax1.bar(
                x_offset,
                rd["Qty"],
                width=bar_width,
                label=f"{region} Units",
                color=revenue_colors[region],
                alpha=0.85,
            )

            for bar, val in zip(bars, rd["Qty"]):
                if val > 0:
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        format_compact(val),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                    )

        ax1.set_xlabel("Month", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Units Sold (Qty)", fontsize=12, fontweight="bold")
        ax1.set_xticks([x + bar_width * 1.5 for x in x_pos])
        ax1.set_xticklabels([pd.to_datetime(m).strftime("%b %Y") for m in months_ts], rotation=0, ha="center")
        ax1.grid(True, axis="y", alpha=0.3)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))

        ax2 = ax1.twinx()
        ax2.plot(
            [x + bar_width * 1.5 for x in x_pos],
            monthly_amt["ProfitMargin"],
            marker="o",
            label="Profit Margin (%)",
            color="#2E8B57",
            linewidth=3,
            markersize=8,
        )
        ax2.set_ylabel("Profit Margin (%)", fontsize=12, fontweight="bold")
        ax2.set_ylim(-100, 100)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, ncol=2)

        plt.title(
            f"Units Sold and Profit Margin by Region per Month (Cost Method: {cost_method})",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    # =========================================================
    # AMOUNT MODE
    # =========================================================
    if view_type == "Region":
        region_summary = (
            Sales_temp.groupby("Region", dropna=False)
            .agg({"Revenue": "sum", "TotalCost": "sum", "NetProfit": "sum"})
            .reset_index()
        )
        region_summary["ProfitMargin"] = safe_margin_pct(region_summary["NetProfit"], region_summary["Revenue"])
        region_summary["Region"] = pd.Categorical(region_summary["Region"], categories=region_order, ordered=True)
        region_summary = region_summary.sort_values("Region")

        fig, ax1 = plt.subplots(figsize=(18, 8))
        bar_width = 0.2
        x_pos = list(range(len(region_order)))

        for i, region in enumerate(region_order):
            rd = region_summary[region_summary["Region"] == region]
            if len(rd) == 0:
                continue

            revenue_val = float(rd["Revenue"].values[0])
            cost_val = float(rd["TotalCost"].values[0])

            ax1.bar(
                i,
                revenue_val / 1e6,
                width=bar_width * 2,
                label=f"{region} Revenue" if i == 0 else "",
                color=revenue_colors[region],
                alpha=0.8,
            )
            if revenue_val > 0:
                ax1.text(
                    i,
                    revenue_val / 1e6,
                    format_compact(revenue_val),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color=revenue_colors[region],
                )

            ax1.bar(
                i,
                -cost_val / 1e6,
                width=bar_width * 2,
                label=f"{region} Cost" if i == 0 else "",
                color=cost_colors[region],
                alpha=0.7,
            )
            if cost_val > 0:
                ax1.text(
                    i,
                    -cost_val / 1e6,
                    format_compact(cost_val),
                    ha="center",
                    va="top",
                    fontsize=9,
                    fontweight="bold",
                    color=cost_colors[region],
                )

        ax1.axhline(y=0, color="black", linewidth=1.5)
        ax1.set_xlabel("Region", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Revenue / Cost (Millions $)", fontsize=12, fontweight="bold")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(region_order)
        ax1.grid(True, axis="y", alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(
            x_pos,
            region_summary["ProfitMargin"],
            marker="o",
            label="Profit Margin (%)",
            color="#2E8B57",
            linewidth=3,
            markersize=8,
        )
        ax2.set_ylabel("Profit Margin (%)", fontsize=12, fontweight="bold")
        ax2.set_ylim(-100, 100)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, ncol=2)

        plt.title(
            f"Revenue, Cost, and Profit Margin by Region (Cost Method: {cost_method})",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    # Amount Month view
    region_month_summary = (
        Sales_temp.groupby(["Region", "YearMonth"], dropna=False)
        .agg({"Revenue": "sum", "TotalCost": "sum", "NetProfit": "sum"})
        .reset_index()
    )
    region_month_summary["YearMonth"] = region_month_summary["YearMonth"].dt.to_timestamp()

    monthly_summary = (
        Sales_temp.groupby("YearMonth", dropna=False)
        .agg({"Revenue": "sum", "TotalCost": "sum", "NetProfit": "sum"})
        .reset_index()
    )
    monthly_summary["YearMonth"] = monthly_summary["YearMonth"].dt.to_timestamp()
    monthly_summary = monthly_summary.set_index("YearMonth").reindex(months_ts, fill_value=0).reset_index()
    monthly_summary.rename(columns={"index": "YearMonth"}, inplace=True)
    monthly_summary["ProfitMargin"] = safe_margin_pct(monthly_summary["NetProfit"], monthly_summary["Revenue"])

    bar_width = 0.2
    x_pos = list(range(len(months_ts)))

    all_combinations = pd.MultiIndex.from_product([region_order, months_ts], names=["Region", "YearMonth"])
    complete_data = (
        region_month_summary.set_index(["Region", "YearMonth"])
        .reindex(all_combinations, fill_value=0)
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(18, 8))

    for i, region in enumerate(region_order):
        rd = complete_data[complete_data["Region"] == region].sort_values("YearMonth")
        x_offset = [x + (i * bar_width) for x in x_pos]

        revenue_bars = ax1.bar(
            x_offset,
            rd["Revenue"] / 1e6,
            width=bar_width,
            label=f"{region} Revenue",
            color=revenue_colors[region],
            alpha=0.8,
        )
        for bar, val in zip(revenue_bars, rd["Revenue"]):
            if val > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    format_compact(val),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    color=revenue_colors[region],
                )

        cost_bars = ax1.bar(
            x_offset,
            -rd["TotalCost"] / 1e6,
            width=bar_width,
            label=f"{region} Cost",
            color=cost_colors[region],
            alpha=0.7,
        )
        for bar, val in zip(cost_bars, rd["TotalCost"]):
            if val > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    format_compact(val),
                    ha="center",
                    va="top",
                    fontsize=7,
                    fontweight="bold",
                    color=cost_colors[region],
                )

    ax1.axhline(y=0, color="black", linewidth=1.5)
    ax1.set_xlabel("Month", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Revenue / Cost (Millions $)", fontsize=12, fontweight="bold")
    ax1.set_xticks([x + bar_width * 1.5 for x in x_pos])
    ax1.set_xticklabels([pd.to_datetime(m).strftime("%b %Y") for m in months_ts], rotation=0, ha="center")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        [x + bar_width * 1.5 for x in x_pos],
        monthly_summary["ProfitMargin"],
        marker="o",
        label="Profit Margin (%)",
        color="#2E8B57",
        linewidth=3,
        markersize=8,
    )
    ax2.set_ylabel("Profit Margin (%)", fontsize=12, fontweight="bold")
    ax2.set_ylim(-100, 100)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, ncol=2)

    plt.title(
        f"Revenue, Cost, and Profit Margin by Region per Month (Cost Method: {cost_method})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Sales / Cost Dashboard")

st.sidebar.header("Controls")
cost_method = st.sidebar.selectbox("Cost Method", ["Moving Average", "Total Average"])
view_type = st.sidebar.selectbox("View Type", ["Month", "Region"])
metric_type = st.sidebar.selectbox("Metric", ["Amount", "Count"])

st.sidebar.divider()
st.sidebar.subheader("Upload data")
sales_file = st.sidebar.file_uploader("Sales.csv", type=["csv"])
cost_file = st.sidebar.file_uploader("Cost_dated.csv", type=["csv"])

if not sales_file or not cost_file:
    st.info("Upload both Sales.csv and Cost_dated.csv to render the dashboard.")
    st.stop()

Sales_base = pd.read_csv(sales_file)
Cost_dated = pd.read_csv(cost_file)

with st.expander("Preview: Sales.csv"):
    st.dataframe(Sales_base.head(20), use_container_width=True)

with st.expander("Preview: Cost_dated.csv"):
    st.dataframe(Cost_dated.head(20), use_container_width=True)

try:
    fig = create_visualization(
        Sales_base=Sales_base,
        Cost_dated=Cost_dated,
        cost_method=cost_method,
        view_type=view_type,
        metric_type=metric_type,
    )
    st.pyplot(fig, clear_figure=True, use_container_width=True)
except Exception as e:
    st.error(f"Dashboard error: {e}")
