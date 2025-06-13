import pandas as pd
import numpy as np
import altair as alt
import streamlit as st


# --- Load Data ---
df = pd.read_csv("open-llm-leaderboards.csv")



# Clean and preprocess
df.columns = df.columns.str.strip()
df['Submission Date'] = pd.to_datetime(df['Submission Date'], errors='coerce')
df = df[df['Type'].notna()]

# Sidebar filters
st.sidebar.header("Filters")

# Date filter
if df['Submission Date'].notna().any():
    min_date = df['Submission Date'].min()
    max_date = df['Submission Date'].max()
    date_range = st.sidebar.slider("Submission Date Range:",
                                   min_value=min_date.date(),
                                   max_value=max_date.date(),
                                   value=(min_date.date(), max_date.date()))
    df = df[(df['Submission Date'].dt.date >= date_range[0]) & (df['Submission Date'].dt.date <= date_range[1])]

# Type filter
type_options = df["Type"].unique().tolist()
selected_types = st.sidebar.multiselect("Select Model Types:", options=type_options, default=type_options)
df = df[df["Type"].isin(selected_types)]

# Score columns
score_cols = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'Average ⬆️']

# Filter by minimum average score
min_score = st.sidebar.slider("Minimum Average Score",
                              float(df['Average ⬆️'].min()),
                              float(df['Average ⬆️'].max()),
                              float(df['Average ⬆️'].min()))
df = df[df['Average ⬆️'] >= min_score]

# Aggregate
grouped = df.groupby("Type").agg({
    "IFEval": "mean",
    "BBH": "mean",
    "MATH Lvl 5": "mean",
    "GPQA": "mean",
    "MUSR": "mean",
    "Average ⬆️": "mean",
    "Hub ❤️": "mean",
    "CO₂ cost (kg)": "mean",
    "Model": "count"
}).reset_index().rename(columns={"Model": "Model Count"})

st.title("💡 Open LLM Leaderboard — Streamlit Dashboard")



st.markdown("### 📊 LLM Evaluation Metrics Overview")

st.markdown("""
| Metric        | Description                                                                                                                                                                                                                                                                  | Source |
|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| **IFEval**     | Tests whether a model can follow **explicit formatting instructions** (e.g., "use keyword X", "structure Y"). Focus is on **format adherence**, not content quality. This allows for rigorous automatic evaluation.                                                         | [IFEval (2023)](https://arxiv.org/abs/2311.07911) |
| **BBH**        | A hard subset of **23 BigBench tasks**, designed for **objective, challenging, and diverse reasoning tasks** (e.g., multistep math, boolean logic, sarcasm detection). **Correlates well with human preferences** and provides statistically significant insights.       | [BBH (2022)](https://arxiv.org/abs/2210.09261) |
| **MATH Lvl 5** | High-school level **math competition problems**, presented with consistent **LaTeX + Asymptote** formatting. Requires **precise structured responses**. Only includes **Level 5** difficulty problems.                                                                      | [MATH (2021)](https://arxiv.org/abs/2103.03874) |
| **GPQA**       | Expert-authored **graduate-level STEM Q&A** covering biology, physics, chemistry. Carefully validated and **gated for contamination risk**. Designed to test **factual and deep domain knowledge**.                                                                         | [GPQA (2023)](https://arxiv.org/abs/2311.12022) |
| **MuSR**       | Algorithmically generated problems like **murder mysteries** and **complex team optimizations** (~1000 words each). Requires **long-context, multi-step reasoning**. Most models **perform near random**.                                                                 | [MuSR (2023)](https://arxiv.org/abs/2310.16049) |
| **MMLU-Pro**   | Updated version of MMLU. Addresses issues like **data contamination**, **low difficulty**, and **noisy questions**. Includes **10 answer choices**, more reasoning, and **expert-reviewed** items. Considered **harder and cleaner** than original MMLU.                   | [MMLU-Pro (2024)](https://arxiv.org/abs/2406.01574) |
""")



# Transform data for faceted chart
long_df = grouped.melt(
    id_vars=["Type"],
    value_vars=score_cols,
    var_name="Metric",
    value_name="Score"
)

# Get unique model types
types = long_df["Type"].unique().tolist()

# Chunk types into rows of 3
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Build chart for each type and lay out
for row_types in chunks(types, 3):
    cols = st.columns(len(row_types))  # create as many columns as needed
    for i, t in enumerate(row_types):
        chart = alt.Chart(long_df).transform_filter(
            alt.datum.Type == t
        ).encode(
            x=alt.X("Metric:N", title="Evaluation Metric"),
            y=alt.Y("Score:Q", title="Avg Score"),
            color=alt.Color("Metric:N", legend=None)
        )

        composed = alt.layer(
            chart.mark_bar(),
            chart.mark_text(
                align="center",
                baseline="bottom",
                dy=-3,
                fontSize=11
            ).encode(text=alt.Text("Score:Q", format=".2f"))
        ).properties(
            title=t,
            width=220,
            height=300
        )

        cols[i].altair_chart(composed, use_container_width=True)
















# Chart: Model Count by Type
df.columns = df.columns.str.strip()
df['Type'] = df['Type'].astype(str)

# Count the number of entries per Type
type_counts = df['Type'].value_counts().reset_index()
type_counts.columns = ['Type', 'Count']

# Pie chart
pie = alt.Chart(type_counts).mark_arc(innerRadius=50, outerRadius=150).encode(
    theta=alt.Theta(field='Count', type='quantitative'),
    color=alt.Color(field='Type', type='nominal'),
    tooltip=['Type', 'Count']
).properties(
    width=400,
    height=400,
    title='Distribution of Model Types'
)

pie



# --- CLEAN & AGGREGATE ---
df.columns = df.columns.str.strip()
df = df.rename(columns={"Average ⬆️": "Average"})

df = df.dropna(subset=["CO₂ cost (kg)", "Type"])
df["CO₂ cost (kg)"] = pd.to_numeric(df["CO₂ cost (kg)"], errors="coerce")
df = df.dropna(subset=["CO₂ cost (kg)"])

# Aggregate by Type
agg_df = df.groupby("Type", as_index=False)["CO₂ cost (kg)"].sum()
agg_df["CO₂ Rounded"] = agg_df["CO₂ cost (kg)"].round().astype(int)
agg_df["Size"] = agg_df["CO₂ cost (kg)"] ** 4

# --- POSITION CIRCLES IN SPIRAL (SIMULATED PACKING) ---
agg_df = agg_df.sort_values("CO₂ cost (kg)", ascending=False).reset_index(drop=True)

# Polar layout
def polar_positions(n, radius_step=1.5):
    angles = []
    radii = []
    for i in range(n):
        r = radius_step * np.sqrt(i)
        theta = i * 137.5  # Golden angle (degrees)
        angles.append(np.deg2rad(theta))
        radii.append(r)
    x = [r * np.cos(a) for r, a in zip(radii, angles)]
    y = [r * np.sin(a) for r, a in zip(radii, angles)]
    return x, y

agg_df["x"], agg_df["y"] = polar_positions(len(agg_df), radius_step=0.4)


# --- BUBBLE CHART ---
bubbles = alt.Chart(agg_df).mark_circle(opacity=0.8).encode(
    x=alt.X("x:Q", axis=None),
    y=alt.Y("y:Q", axis=None),
    size=alt.Size("Size:Q", scale=alt.Scale(range=[3000, 35000]), legend=None),
    color=alt.Color("Type:N", legend=alt.Legend(title="Model Type")),
    tooltip=[
        alt.Tooltip("Type:N", title="Model Type"),
        alt.Tooltip("CO₂ cost (kg):Q", title="Total CO₂ (kg)", format=",.0f")
    ]
).properties(
    title="Packed Bubble Chart of CO₂ Emissions by Model Type",
    width=700,
    height=600
)

# --- LABELS ---
labels = alt.Chart(agg_df).mark_text(
    fontSize=11,
    fontWeight="bold",
    color="black"
).encode(
    x="x:Q",
    y="y:Q",
    text="CO₂ Rounded:Q"
)

# --- COMBINE ---
st.altair_chart(bubbles + labels, use_container_width=True)


# Ensure 'Upload To Hub Date' is datetime
df['Upload To Hub Date'] = pd.to_datetime(df['Upload To Hub Date'], errors='coerce')
df = df.dropna(subset=['Upload To Hub Date', 'CO₂ cost (kg)', 'Type'])

# Extract month
df['Month'] = df['Upload To Hub Date'].dt.to_period('M').dt.to_timestamp()

# Group + cumulative CO₂ per Type
monthly_emissions = (
    df.groupby(['Month', 'Type'])['CO₂ cost (kg)']
    .sum()
    .reset_index()
)

monthly_emissions['Cumulative CO₂'] = (
    monthly_emissions.sort_values("Month")
    .groupby('Type')["CO₂ cost (kg)"]
    .cumsum()
)

stacked_area = alt.Chart(monthly_emissions).mark_area(interpolate='monotone').encode(
    x=alt.X("Month:T", title="Month"),
    y=alt.Y("Cumulative CO₂:Q", stack="zero", title="Cumulative CO₂ Emissions (kg)"),
    color=alt.Color("Type:N", title="Model Type"),
    tooltip=["Month:T", "Type:N", "Cumulative CO₂:Q"]
).properties(
    title="Accumulating Carbon Emissions from AI Models Over Time (Stacked by Type)",
    width=800,
    height=800
)

st.altair_chart(stacked_area, use_container_width=True)









df.columns = df.columns.str.strip()
df = df.rename(columns={"Average ⬆️": "Average"})
df = df.dropna(subset=["Hub ❤️", "Average", "Type"])

df["Hub ❤️"] = pd.to_numeric(df["Hub ❤️"], errors="coerce")
df["Average"] = pd.to_numeric(df["Average"], errors="coerce")
df = df.dropna(subset=["Hub ❤️", "Average"])

# --- BASE SCATTER ---
base = alt.Chart(df).encode(
    x=alt.X("Hub ❤️:Q", title="User Satisfaction"),
    y=alt.Y("Average:Q", title="Average Score"),
    color=alt.Color("Type:N", legend=alt.Legend(title="Model Type")),
    tooltip=["Type", "Hub ❤️", "Average"]
)

# --- POINTS ---
points = base.mark_circle(size=90)

# --- REGRESSION LINE ---
trend = base.transform_regression("Hub ❤️", "Average").mark_line(color="black", strokeDash=[4, 4])

# --- MARGINAL HISTOGRAMS ---
x_hist = alt.Chart(df).mark_bar(opacity=0.3).encode(
    x=alt.X("Hub ❤️:Q", bin=True, title="User Satisfaction"),
    y=alt.Y("count():Q", title=None)
).properties(height=80)

y_hist = alt.Chart(df).mark_bar(opacity=0.3).encode(
    x=alt.X("count():Q", title=None),
    y=alt.Y("Average:Q", bin=True, title="Average Score")
).properties(width=80)

# --- COMBINE ---
scatter_with_trend = points + trend
main_chart = alt.hconcat(scatter_with_trend, y_hist)
full_chart = alt.vconcat(x_hist, main_chart).resolve_axis(x='shared', y='shared').properties(
    title="Correlation Between User Satisfaction and Average Score"
)

# --- SHOW ---
st.altair_chart(full_chart, use_container_width=True)


