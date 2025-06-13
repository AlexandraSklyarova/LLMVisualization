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
            width=400,
            height=600
        )

        cols[i].altair_chart(composed, use_container_width=True)





#new 

selected_metric = st.selectbox("Highlight a Metric:", score_cols)

# --- Generate charts by model Type ---
for row_types in chunks(types, 3):
    cols = st.columns(len(row_types))
    for i, t in enumerate(row_types):
        chart_base = alt.Chart(long_df).transform_filter(
            alt.datum.Type == t
        ).encode(
            x=alt.X("Metric:N", title="Evaluation Metric"),
            y=alt.Y("Score:Q", title="Avg Score"),
            color=alt.Color("Metric:N", legend=None),
            opacity=alt.condition(
                f"datum.Metric === '{selected_metric}'",
                alt.value(1.0),
                alt.value(0.2)
            )
        )

        bars = chart_base.mark_bar()
        labels = chart_base.mark_text(
            align="center",
            baseline="bottom",
            dy=-3,
            fontSize=11
        ).encode(text=alt.Text("Score:Q", format=".2f"))

        composed = alt.layer(bars, labels).properties(
            title=t,
            width=400,
            height=600
        )

        cols[i].altair_chart(composed, use_container_width=True)





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
    size=alt.Size("Size:Q", scale=alt.Scale(range=[2500, 30000]), legend=None),
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

# Group by month and type
monthly_emissions = (
    df.groupby(['Month', 'Type'])['CO₂ cost (kg)']
    .sum()
    .reset_index()
)

# Cumulative emissions
monthly_emissions['Cumulative CO₂'] = (
    monthly_emissions.sort_values("Month")
    .groupby('Type')["CO₂ cost (kg)"]
    .cumsum()
)

# Main stacked area chart
stacked_area = alt.Chart(monthly_emissions).mark_area(interpolate='monotone').encode(
    x=alt.X("Month:T", title="Month", axis=alt.Axis(format="%b %Y")),  # e.g., Jan 2024
    y=alt.Y("Cumulative CO₂:Q", stack="zero", title="Cumulative CO₂ Emissions (kg)"),
    color=alt.Color("Type:N", title="Model Type"),
    tooltip=[
        alt.Tooltip("Month:T", title="Month", format="%B %Y"),  # e.g., January 2024
        alt.Tooltip("Type:N", title="Model Type"),
        alt.Tooltip("Cumulative CO₂:Q", title="Cumulative CO₂ (kg)", format=",.0f")
    ]
)

# Dashed vertical lines for each January
year_lines = alt.Chart(pd.DataFrame({
    "year": pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01"])
})).mark_rule(
    strokeDash=[6, 4],
    color="gray"
).encode(
    x=alt.X("year:T")
)

# Combine and display
final_chart = (stacked_area + year_lines).properties(
    title="Accumulating Carbon Emissions from AI Models Over Time (Stacked by Type)",
    width=1500,
    height=800
)

st.altair_chart(final_chart, use_container_width=True)




#new

df.columns = df.columns.str.strip()
df = df.rename(columns={"Average ⬆️": "Average"})

df = df.dropna(subset=["CO₂ cost (kg)", "Type", "Upload To Hub Date"])
df["CO₂ cost (kg)"] = pd.to_numeric(df["CO₂ cost (kg)"], errors="coerce")
df["Upload To Hub Date"] = pd.to_datetime(df["Upload To Hub Date"], errors="coerce")
df = df.dropna(subset=["CO₂ cost (kg)", "Upload To Hub Date"])
df["Month"] = df["Upload To Hub Date"].dt.to_period('M').dt.to_timestamp()

# --- Create selection from the legend ---
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
    size=alt.Size("Size:Q", scale=alt.Scale(range=[2500, 30000]), legend=None),
    color=alt.Color("Type:N", legend=alt.Legend(title="Model Type")),
    opacity=alt.condition(type_selection, alt.value(1.0), alt.value(0.2)),
    tooltip=[
        alt.Tooltip("Type:N", title="Model Type"),
        alt.Tooltip("CO₂ cost (kg):Q", title="Total CO₂ (kg)", format=",.0f")
    ]
).add_params(type_selection).properties(
    title="Packed Bubble Chart of CO₂ Emissions by Model Type",
    width=1000,
    height=600
)

# --- Labels on bubbles ---
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
st.altair_chart(bubble + labels, use_container_width=True)


# --- Area chart data ---
monthly = df.groupby(["Month", "Type"])["CO₂ cost (kg)"].sum().reset_index()
monthly["Cumulative CO₂"] = monthly.sort_values("Month").groupby("Type")["CO₂ cost (kg)"].cumsum()

# --- Area chart ---
area = alt.Chart(monthly).mark_area(interpolate="monotone").encode(
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
    width=1000,
    height=500
)

st.altair_chart(area, use_container_width=True)





