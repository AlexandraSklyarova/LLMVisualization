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







# Transform data for faceted chart 

long_df = grouped.melt(
    id_vars=["Type"],
    value_vars=score_cols,
    var_name="Metric",
    value_name="Score"
)

# --- Shared selection ---
metric_selection = alt.selection_point(fields=["Metric"], bind="legend")  # optional bind

# --- Base bar chart ---
base = alt.Chart(long_df).mark_bar().encode(
    x=alt.X("Metric:N", title="Evaluation Metric"),
    y=alt.Y("Score:Q", title="Average Score"),
    color=alt.Color("Metric:N", legend=alt.Legend(title="Select Metric")),
    opacity=alt.condition(metric_selection, alt.value(1.0), alt.value(0.2)),
    tooltip=["Type:N", "Metric:N", alt.Tooltip("Score:Q", format=".2f")]
).add_params(metric_selection)

# --- Add labels ---
labels = alt.Chart(long_df).mark_text(
    align="center",
    baseline="bottom",
    dy=-5,
    fontSize=11
).encode(
    x="Metric:N",
    y="Score:Q",
    text=alt.Text("Score:Q", format=".2f"),
    opacity=alt.condition(metric_selection, alt.value(1.0), alt.value(0.2))
)

# --- Combine bar and text, then facet by Type ---
chart = (base + labels).facet(
    column=alt.Column("Type:N", title=None, header=alt.Header(labelAngle=0))
).resolve_scale(
    y="independent"
).properties(
    title="Scores by Evaluation Metric (Click Metric in legend to Highlight Across All Types)"
)

# --- Display in Streamlit ---
st.altair_chart(chart, use_container_width=True)







st.markdown("###  LLM Evaluation Metrics Overview")

evaluation_summary = {
    "IFEval": {
        "Description": "Tests if a model can follow explicit formatting instructions (e.g., include keyword X, use format Y). Focus is on format adherence."
    },
    "BBH": {
        "Description": "Challenging reasoning benchmark of 23 BigBench tasks (math, logic, language). Correlates with human judgment."
    },
    "MATH Lvl 5": {
        "Description": "Level 5 high school math competition problems. Requires exact output format using LaTeX/Asymptote."
    },
    "GPQA": {
        "Description": "Graduate-level STEM questions validated by experts (biology, chemistry, physics). Gated to avoid contamination."
    },
    "MuSR": {
        "Description": "Long, multistep reasoning problems (e.g., mysteries, logistics). Requires long-context understanding."
    },
    "MMLU-Pro": {
        "Description": "Refined version of MMLU with 10 choices, higher difficulty, cleaner data, and expert review."
    }
}

# Reformat into a transposed DataFrame
evaluation_df = pd.DataFrame.from_dict(evaluation_summary, orient="columns")
evaluation_df.index.name = "Info"

# Show the table
st.table(evaluation_df)



# --- SAFELY HANDLE METRIC COLUMNS ---
# First make sure all expected metric columns are actually in the grouped dataframe
expected_metrics = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MuSR', 'MMLU-Pro']
score_cols = [col for col in expected_metrics if col in grouped.columns]

# Get best model type for each metric
best_types = []
for metric in score_cols:
    best_row = grouped.loc[grouped[metric].idxmax()]
    best_types.append({
        "Metric": metric,
        "Best Model Type": best_row["Type"],
        "Average Score": round(best_row[metric], 2)
    })

# Create and display table
best_df = pd.DataFrame(best_types)
st.markdown("### 🏆 Best LLM Type per Evaluation Metric")
st.table(best_df)












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





df['Upload To Hub Date'] = pd.to_datetime(df['Upload To Hub Date'], errors='coerce')
df = df.dropna(subset=['Upload To Hub Date', 'Type'])

# Extract month
df['Month'] = df['Upload To Hub Date'].dt.to_period('M').dt.to_timestamp()

# Count models per month per type
monthly_counts = (
    df.groupby(['Month', 'Type'])
    .size()
    .reset_index(name='Model Count')
)

# Compute cumulative count per type
monthly_counts['Cumulative Models'] = (
    monthly_counts.sort_values('Month')
    .groupby('Type')['Model Count']
    .cumsum()
)

# Main line chart
# Main line chart with formatted x-axis and tooltip
line_chart = alt.Chart(monthly_counts).mark_line().encode(
    x=alt.X("Month:T", title="Month", axis=alt.Axis(format="%b %Y")),  # e.g., Jan 2024
    y=alt.Y("Cumulative Models:Q", title="Total Number of Models"),
    color=alt.Color("Type:N", title="Model Type"),
    tooltip=[
        alt.Tooltip("Month:T", title="Month", format="%B %Y"),  # e.g., January 2024
        alt.Tooltip("Type:N", title="Model Type"),
        alt.Tooltip("Cumulative Models:Q", title="Cumulative Models", format=",.0f")
    ]
)

# Vertical dashed annotation line for April 2025
event_date = pd.to_datetime("2024-04-01")
event_rule = alt.Chart(pd.DataFrame({"date": [event_date]})).mark_rule(
    strokeDash=[4, 4],
    color="red"
).encode(
    x=alt.X("date:T")
)

# Text label for the event
event_text = alt.Chart(pd.DataFrame({
    "date": [event_date],
    "label": ["Publication of Visualization-of-Thought (VoT)"]
})).mark_text(
    align="left",
    baseline="top",
    dx=5,
    dy=-5,
    fontSize=11,
    fontStyle="italic",
    color="red"
).encode(
    x="date:T",
    y=alt.value(0),  # anchor at bottom
    text="label:N"
)

# Combine everything
final_chart = (line_chart + event_rule + event_text).properties(
    title="Cumulative Number of LLM Models Released Over Time",
    width=1100,
    height=500
)

st.altair_chart(final_chart, use_container_width=True)












df.columns = df.columns.str.strip()
df = df.rename(columns={"Average ⬆️": "Average"})

df = df.dropna(subset=["CO₂ cost (kg)", "Type", "Upload To Hub Date"])
df["CO₂ cost (kg)"] = pd.to_numeric(df["CO₂ cost (kg)"], errors="coerce")
df["Upload To Hub Date"] = pd.to_datetime(df["Upload To Hub Date"], errors="coerce")
df = df.dropna(subset=["CO₂ cost (kg)", "Upload To Hub Date"])
df["Month"] = df["Upload To Hub Date"].dt.to_period('M').dt.to_timestamp()

# --- Selection shared across both charts ---
type_selection = alt.selection_point(fields=["Type"], bind="legend")

# --- Bubble chart data ---
bubble_data = df.groupby("Type", as_index=False)["CO₂ cost (kg)"].sum()
bubble_data["CO₂ Rounded"] = bubble_data["CO₂ cost (kg)"].round().astype(int)
bubble_data["Size"] = bubble_data["CO₂ cost (kg)"] ** 4

def polar_positions(n, radius_step=0.4):
    angles, radii = [], []
    for i in range(n):
        r = radius_step * np.sqrt(i)
        theta = i * 137.5
        angles.append(np.deg2rad(theta))
        radii.append(r)
    x = [r * np.cos(a) for r, a in zip(radii, angles)]
    y = [r * np.sin(a) for r, a in zip(radii, angles)]
    return x, y

bubble_data["x"], bubble_data["y"] = polar_positions(len(bubble_data))

# --- Bubble chart ---
bubbles = alt.Chart(bubble_data).mark_circle(opacity=0.9).encode(
    x=alt.X("x:Q", axis=None),
    y=alt.Y("y:Q", axis=None),
    size=alt.Size("Size:Q", scale=alt.Scale(range=[3500, 35000]), legend=None),
    color=alt.Color("Type:N", legend=alt.Legend(title="Model Type")),
    opacity=alt.condition(type_selection, alt.value(1.0), alt.value(0.2)),
    tooltip=[
        alt.Tooltip("Type:N", title="Model Type"),
        alt.Tooltip("CO₂ cost (kg):Q", title="Total CO₂ (kg)", format=",.0f")
    ]
).add_params(type_selection).properties(
    title="Packed Bubble Chart of CO₂ Emissions by Model Type (Click on Legend to Highlight area in linked CO2 Charts)",
    width=1000,
    height=600
)

labels = alt.Chart(bubble_data).mark_text(
    fontSize=11,
    fontWeight="bold",
    color="black"
).encode(
    x="x:Q",
    y="y:Q",
    text="CO₂ Rounded:Q",
    opacity=alt.condition(type_selection, alt.value(1.0), alt.value(0.2))
)

bubble_chart = bubbles + labels

# --- Area chart data ---
monthly = df.groupby(["Month", "Type"])["CO₂ cost (kg)"].sum().reset_index()
monthly["Cumulative CO₂"] = monthly.sort_values("Month").groupby("Type")["CO₂ cost (kg)"].cumsum()

# --- Area chart ---
area_chart = alt.Chart(monthly).mark_area(interpolate="monotone").encode(
    x=alt.X("Month:T", title="Month", axis=alt.Axis(format="%b %Y")),
    y=alt.Y("Cumulative CO₂:Q", title="Cumulative CO₂ Emissions (kg)", stack="zero"),
    color=alt.Color("Type:N", legend=None),  # suppress duplicate legend
    opacity=alt.condition(type_selection, alt.value(1.0), alt.value(0.2)),
    tooltip=[
        alt.Tooltip("Month:T", title="Month", format="%B %Y"),
        alt.Tooltip("Type:N", title="Model Type"),
        alt.Tooltip("Cumulative CO₂:Q", format=",.0f", title="Cumulative CO₂ (kg)")
    ]
).add_params(type_selection).properties(
    title="Cumulative Carbon Emissions Over Time (Stacked by Type)",
    width=1200,
    height=500
)

# --- Combine both charts vertically ---
combined_chart = alt.vconcat(bubble_chart, area_chart).resolve_legend(color="shared")
st.altair_chart(combined_chart, use_container_width=True)


heatmap = alt.Chart(grouped).transform_bin(
    "binned_satisfaction", field="Hub ❤️", bin=alt.Bin(maxbins=40)
).transform_bin(
    "binned_score", field="Average ⬆️", bin=alt.Bin(maxbins=40)
).transform_aggregate(
    count="count()", groupby=["binned_satisfaction", "binned_score"]
).mark_rect().encode(
    x=alt.X("binned_satisfaction:Q", title="User Satisfaction (Hub ❤️)"),
    y=alt.Y("binned_score:Q", title="Average Score"),
    color=alt.Color("count:Q", scale=alt.Scale(scheme="reds"), title="Density"),
    tooltip=["binned_satisfaction", "binned_score", "count"]
).properties(
    title="Density of Models by User Satisfaction vs Average Score",
    width=600,
    height=400
)

# Display the chart
st.altair_chart(heatmap, use_container_width=True)






