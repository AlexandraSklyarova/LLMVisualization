import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import plotly.express as px
import circlify



# --- Load Data --- 
df = pd.read_csv("open-llm-leaderboards.csv")

 

# Clean and preprocess
df.columns = df.columns.str.strip()
df['Upload To Hub Date'] = pd.to_datetime(df['Upload To Hub Date'], errors='coerce')
df = df[df['Type'].notna()]

# Sidebar filters
st.sidebar.header("Filters")

# Date filter
if df['Upload To Hub Date'].notna().any():
    min_date = df['Upload To Hub Date'].min()
    max_date = df['Upload To Hub Date'].max()
    date_range = st.sidebar.slider("Upload To Hub Date:",
                                   min_value=min_date.date(),
                                   max_value=max_date.date(),
                                   value=(min_date.date(), max_date.date()))
    df = df[(df['Upload To Hub Date'].dt.date >= date_range[0]) & (df['Upload To Hub Date'].dt.date <= date_range[1])]

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

st.title("Efficacy, Enjoyment, and Environment: An Exploration of Open LLM Leaderboards")






st.header("Evaluation of Different LLM Models")

st.markdown("In this section, you can explore the abilities of different LLM types based on various metrics (explained below). Use the sidebar to filter any additional information like how the scores have changed over time. ")

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
    y=alt.Y("Score:Q", title="Average Score", scale=alt.Scale(domain=[0, 55])),
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
).properties(
    title="Scores by Evaluation Metric (Click Metric in legend to Highlight Across All Types)",
    spacing=60  # ✅ Apply spacing here
).resolve_scale(
    y="shared"  # If you want the y-axis consistent
)


# --- Display in Streamlit ---
st.altair_chart(chart, use_container_width=True)


evaluation_summary = {
    "IFEval": "Tests if a model can follow explicit formatting instructions (e.g., include keyword X, use format Y). Focus is on format adherence.",
    "BBH": "Challenging reasoning benchmark of 23 BigBench tasks (math, logic, language). Correlates with human judgment.",
    "MATH Lvl 5": "Level 5 high school math competition problems. Requires exact output format using LaTeX/Asymptote.",
    "GPQA": "Graduate-level STEM questions validated by experts (biology, chemistry, physics). Gated to avoid contamination.",
    "MuSR": "Long, multistep reasoning problems (e.g., mysteries, logistics). Requires long-context understanding.",
    "MMLU-Pro": "Refined version of MMLU with 10 choices, higher difficulty, cleaner data, and expert review."
}

# Create two groups
group1 = ["IFEval", "BBH", "MATH Lvl 5"]
group2 = ["GPQA", "MuSR", "MMLU-Pro"]

# HTML table
html = """
<style>
table {
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
}
td, th {
    border: 1px solid #ddd;
    padding: 12px;
    text-align: left;
    vertical-align: top;
    word-wrap: break-word;
}
tr:nth-child(even) {background-color: #f9f9f9;}
</style>

<table>
<tr>
  <th>{}</th><th>{}</th><th>{}</th>
</tr>
<tr>
  <td>{}</td><td>{}</td><td>{}</td>
</tr>
<tr>
  <th>{}</th><th>{}</th><th>{}</th>
</tr>
<tr>
  <td>{}</td><td>{}</td><td>{}</td>
</tr>
</table>
""".format(
    group1[0], group1[1], group1[2],
    evaluation_summary[group1[0]], evaluation_summary[group1[1]], evaluation_summary[group1[2]],
    group2[0], group2[1], group2[2],
    evaluation_summary[group2[0]], evaluation_summary[group2[1]], evaluation_summary[group2[2]]
)

# Render in Streamlit
st.markdown("### Evaluation Metric Descriptions", unsafe_allow_html=True)
st.markdown(html, unsafe_allow_html=True)


st.markdown("### Evaluation Metrics Overview")

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
        "Description": "Refined version of MMLU, higher difficulty, cleaner data, and expert review."
    }
}

# Break into two lists
metrics1 = ["IFEval", "BBH", "MATH Lvl 5"]
metrics2 = ["GPQA", "MuSR", "MMLU-Pro"]

# Build the 4-row DataFrame
custom_table = pd.DataFrame([
    metrics1,
    [evaluation_summary[m]["Description"] for m in metrics1],
    metrics2,
    [evaluation_summary[m]["Description"] for m in metrics2]
])

# Display the table
st.markdown("### Evaluation Metrics Summary")
st.dataframe(custom_table, use_container_width=True)


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
st.markdown("### Best LLM Type per Evaluation Metric")
st.table(best_df)



st.markdown("---")

st.header("Number of Models and Growth Over Time")

st.markdown("Here you can explore which model types are the most common and how the total number has grown over the years")







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





# --- Prepare data ---
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

# --- Semantic zoom (bind scales) ---
zoom = alt.selection_interval(bind='scales')

# --- Main line chart ---
line_chart = alt.Chart(monthly_counts).mark_line().encode(
    x=alt.X("Month:T", title="Month", axis=alt.Axis(format="%b %Y")),
    y=alt.Y("Cumulative Models:Q", title="Total Number of Models"),
    color=alt.Color("Type:N", title="Model Type"),
    tooltip=[
        alt.Tooltip("Month:T", title="Month", format="%B %Y"),
        alt.Tooltip("Type:N", title="Model Type"),
        alt.Tooltip("Cumulative Models:Q", title="Cumulative Models", format=",.0f")
    ]
).add_params(zoom)

# --- Annotation line for April 2024 ---
event_date = pd.to_datetime("2024-04-01")
event_df = pd.DataFrame({
    "date": [event_date],
    "label": ["Publication of Visualization-of-Thought (VoT)"]
})

event_rule = alt.Chart(event_df).mark_rule(
    strokeDash=[4, 4],
    color="red"
).encode(
    x="date:T"
)

event_text = alt.Chart(event_df).mark_text(
    align="left",
    baseline="top",
    dx=5,
    dy=-5,
    fontSize=11,
    fontStyle="italic",
    color="red"
).encode(
    x="date:T",
    y=alt.value(0),
    text="label:N"
)

# --- Combine everything ---
final_chart = alt.layer(line_chart, event_rule, event_text).properties(
    title="Cumulative Number of LLM Models Released Over Time (Zoom Enabled)",
    width=1100,
    height=500
)

st.altair_chart(final_chart, use_container_width=True)


st.markdown("---")



st.header("CO₂ Emissions Overview")

st.markdown("Here's a breakdown of average CO₂ emissions per model type. Use the legend to filter.")





#new new

df.columns = df.columns.str.strip()
df = df.rename(columns={"Average ⬆️": "Average"})
df["CO₂ cost (kg)"] = pd.to_numeric(df["CO₂ cost (kg)"], errors="coerce")
df["Upload To Hub Date"] = pd.to_datetime(df["Upload To Hub Date"], errors="coerce")
df = df.dropna(subset=["CO₂ cost (kg)", "Upload To Hub Date", "Type"])

# --- Use CO₂ as radius directly ---
grouped = df.groupby("Type", as_index=False)["CO₂ cost (kg)"].mean()

angle_step = 2 * np.pi / len(grouped)
radius = 750  # adjust for spacing

layout_df = grouped.copy()
layout_df["angle"] = [i * angle_step for i in range(len(grouped))]
layout_df["x"] = np.cos(layout_df["angle"]) * radius
layout_df["y"] = np.sin(layout_df["angle"]) * radius

layout_df["r"] = layout_df["CO₂ cost (kg)"]
layout_df["Size"] = (layout_df["r"] ** 2) * np.pi
layout_df["CO₂ Rounded"] = layout_df["r"].round(1)

# Circlify layout using CO₂ directly
circles = circlify.circlify(
    grouped["CO₂ cost (kg)"].tolist(),
    show_enclosure=False,
    target_enclosure=circlify.Circle(x=0, y=0, r=1)
)

# Position and size bubbles
layout_df = pd.DataFrame([{
    "x": c.x * grouped.iloc[i]["CO₂ cost (kg)"] * 1.5 * 50,  # x scaled to radius
    "y": c.y * grouped.iloc[i]["CO₂ cost (kg)"] * 1.5 * 50,  # y scaled to radius
    "r": grouped.iloc[i]["CO₂ cost (kg)"],
    "Type": grouped.iloc[i]["Type"],
    "CO₂ cost (kg)": grouped.iloc[i]["CO₂ cost (kg)"]
} for i, c in enumerate(circles)])


# Size = π × r², to match Altair's area encoding
layout_df["Size"] = layout_df["r"] ** 2 * np.pi
layout_df["CO₂ Rounded"] = layout_df["CO₂ cost (kg)"].round(1)

# Shared selection
type_selection = alt.selection_point(fields=["Type"], bind="legend")

# --- Bubble chart ---
bubbles = alt.Chart(layout_df).mark_circle(opacity=0.85).encode(
    x=alt.X("x:Q", axis=None),
    y=alt.Y("y:Q", axis=None),
    size=alt.Size("Size:Q", scale=alt.Scale(range=[500, 25000]), legend=None),
    color=alt.Color("Type:N", legend=alt.Legend(title="Model Type")),
    opacity=alt.condition(type_selection, alt.value(1.0), alt.value(0.2)),
    tooltip=["Type:N", "CO₂ cost (kg):Q"]
).add_params(type_selection).properties(
    title="Packed Bubble Chart: Radius = CO₂ Cost (kg)",
    width=800,
    height=650
)

labels = alt.Chart(layout_df).mark_text(
    fontSize=13,
    fontWeight="bold",
    color="black"
).encode(
    x="x:Q",
    y="y:Q",
    text="CO₂ Rounded:Q",
    opacity=alt.condition(type_selection, alt.value(1.0), alt.value(0.3))
)

bubble_chart = bubbles + labels

# --- Area chart data ---
df["Month"] = df["Upload To Hub Date"].dt.to_period("M").dt.to_timestamp()
monthly = df.groupby(["Month", "Type"])["CO₂ cost (kg)"].sum().reset_index()
monthly["Cumulative CO₂"] = monthly.sort_values("Month").groupby("Type")["CO₂ cost (kg)"].cumsum()

# --- Area chart ---
zoom = alt.selection_interval(bind="scales")

# --- Area chart with zoom + month-year x-axis formatting ---
area_chart = alt.Chart(monthly).mark_area(interpolate="monotone").encode(
    x=alt.X("Month:T", title="Month", axis=alt.Axis(format="%b %Y")),
    y=alt.Y("Cumulative CO₂:Q", title="Cumulative CO₂ Emissions (kg)", stack="zero"),
    color=alt.Color("Type:N", legend=None),
    opacity=alt.condition(type_selection, alt.value(1.0), alt.value(0.1)),
    tooltip=[
        alt.Tooltip("Month:T", title="Month", format="%B %Y"),
        alt.Tooltip("Type:N"),
        alt.Tooltip("Cumulative CO₂:Q", format=",.0f")
    ]
).add_params(type_selection, zoom).properties(
    title="Cumulative CO₂ Emissions Over Time (Zoom Enabled)",
    width=800,
    height=400
)

# --- Combine vertically ---
combined_chart = alt.vconcat(
    bubble_chart,
    area_chart
).resolve_legend(color="shared")

# --- Show in Streamlit ---
st.altair_chart(combined_chart, use_container_width=True)



st.markdown("---")


st.header("Relationship between Hub Likes and Average Scores of LLMs")

st.markdown("Here you can explore how popular models are in relation to how they perform across different metrics. Filtering by time in the sidebar can reveal more about their relationship")







# --- Clean and Prepare Data ---
df.columns = df.columns.str.strip()
df = df.rename(columns={"Average ⬆️": "Average"})

# Ensure numeric and drop invalid rows
df = df.dropna(subset=["Hub ❤️", "Average", "eval_name"])
df["Hub ❤️"] = pd.to_numeric(df["Hub ❤️"], errors="coerce")
df["Average"] = pd.to_numeric(df["Average"], errors="coerce")
df = df.dropna(subset=["Hub ❤️", "Average"])

# --- Binning Average Score into 5-point intervals ---
df["Average_Bin"] = ((df["Average"] // 5) * 5).astype(int)

# --- Group and Aggregate ---
binned_avg = df.groupby("Average_Bin", as_index=False).agg(
    Mean_Hub_Score=("Hub ❤️", "mean"),
    Eval_Count=("eval_name", "count")
)

# Optional: Fill any remaining NaNs just in case
binned_avg["Mean_Hub_Score"] = binned_avg["Mean_Hub_Score"].fillna(0)
binned_avg["Eval_Count"] = binned_avg["Eval_Count"].fillna(0).astype(int)

# --- Brushing Selection ---
brush = alt.selection_interval(encodings=["x"])

# --- Heatmap-style Bar Chart ---
heatmap = alt.Chart(binned_avg).mark_bar().encode(
    x=alt.X("Average_Bin:O", title="Average Score Bin (5 pt range)"),
    y=alt.Y("Mean_Hub_Score:Q", title="Mean User Satisfaction", scale=alt.Scale(domain=[0, 100])),
    color=alt.condition(
        brush,
        alt.Color("Mean_Hub_Score:Q", scale=alt.Scale(scheme="blues"), title="Mean Hub ❤️"),
        alt.value("lightgray")
    ),
    tooltip=[
        alt.Tooltip("Average_Bin:O", title="Average Score Bin"),
        alt.Tooltip("Mean_Hub_Score:Q", title="Mean Hub ❤️", format=".1f"),
        alt.Tooltip("Eval_Count:Q", title="Number of Models")
    ]
).add_params(
    brush
).properties(
    title="Mean User Satisfaction by Average Score Bin (Brush to Highlight)",
    width=600,
    height=400
)

st.altair_chart(heatmap, use_container_width=True)

