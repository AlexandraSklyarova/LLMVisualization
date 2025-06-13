import streamlit as st
import pandas as pd
import altair as alt

# Set wide layout
st.set_page_config(layout="wide", page_title="LLM Leaderboard Dashboard")

# Load data
df = pd.read_csv("open-llm-leaderboards.csv")

# Clean and preprocess
df.columns = df.columns.str.strip()
df['Submission Date'] = pd.to_datetime(df['Submission Date'], errors='coerce')

# Filter out missing types or scores
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

# Sidebar: Minimum Average Score
min_score = st.sidebar.slider("Minimum Average Score", float(df['Average ⬆️'].min()), float(df['Average ⬆️'].max()), float(df['Average ⬆️'].min()))
df = df[df['Average ⬆️'] >= min_score]

# Aggregated by Type
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

# Bar Chart: Average Scores by Type
score_chart = alt.Chart(grouped).transform_fold(
    score_cols,
    as_=["Metric", "Score"]
).mark_bar().encode(
    x=alt.X("Metric:N", title="Evaluation Metric"),
    y=alt.Y("Score:Q", title="Average Score"),
    color=alt.Color("Type:N"),
    column=alt.Column("Type:N", title="Model Type")
).properties(title="Evaluation Metrics by Model Type").interactive()

st.altair_chart(score_chart, use_container_width=True)

# Chart: CO₂ Emissions vs Average Score
scatter = alt.Chart(grouped).mark_circle(size=120).encode(
    x=alt.X("CO₂ cost (kg):Q", title="CO₂ Emissions (kg)"),
    y=alt.Y("Average ⬆️:Q", title="Average Score"),
    color="Type:N",
    tooltip=["Type", "Average ⬆️", "CO₂ cost (kg)", "Model Count"]
).properties(title="Model Type Efficiency (Score vs CO₂)").interactive()

st.altair_chart(scatter, use_container_width=True)

# Chart: User Satisfaction vs Score
satisfaction = alt.Chart(grouped).mark_circle(size=120).encode(
    x=alt.X("Hub ❤️:Q", title="User Satisfaction (Hub ❤️)"),
    y=alt.Y("Average ⬆️:Q", title="Average Score"),
    color="Type:N",
    tooltip=["Type", "Hub ❤️", "Average ⬆️"]
).properties(title="User Satisfaction vs Average Score").interactive()

st.altair_chart(satisfaction, use_container_width=True)

# Chart: Model Count by Type
df.columns = df.columns.str.strip()
df['Type'] = df['Type'].astype(str)

# Count the number of entries per Type
type_counts = df['Type'].value_counts().reset_index()
type_counts.columns = ['Type', 'Count']

# Pie chart
pie = alt.Chart(type_counts).mark_arc(innerRadius=50, outerRadius=100).encode(
    theta=alt.Theta(field='Count', type='quantitative'),
    color=alt.Color(field='Type', type='nominal'),
    tooltip=['Type', 'Count']
).properties(
    width=400,
    height=400,
    title='Distribution of Model Types'
)

pie

df.columns = df.columns.str.strip()
df['Submission Date'] = pd.to_datetime(df['Submission Date'], errors='coerce')
df = df[df['Type'].notna() & df['Average ⬆️'].notna()]

# Optional: downsample to reduce visual clutter
df = df.sort_values('Submission Date')

# Line chart of average scores over time per Type
line_chart = alt.Chart(df).mark_line(point=True).encode(
    x=alt.X("Submission Date:T", title="Date"),
    y=alt.Y("Average ⬆️:Q", title="Average Score"),
    color=alt.Color("Type:N"),
    tooltip=["Model", "Type", "Average ⬆️", "Submission Date", "CO₂ cost (kg)", "Hub ❤️"]
).properties(
    title="📈 Evolution of LLM Performance Over Time by Model Type"
)

# Annotations: define key events manually
annotations = pd.DataFrame([
    {"date": "2023-05-01", "note": "Instruction-tuned models begin dominating."},
    {"date": "2023-11-15", "note": "CO₂ cost drops for chat models while scores remain high."},
    {"date": "2024-03-01", "note": "Base models decline in user satisfaction."},
    {"date": "2024-07-01", "note": "New wave of high-performing 'chat' models"},
])
annotations["date"] = pd.to_datetime(annotations["date"])

annotation_layer = alt.Chart(annotations).mark_rule(color="gray", strokeDash=[4,2]).encode(
    x="date:T"
).properties()

text_layer = alt.Chart(annotations).mark_text(align="left", dx=5, dy=-5, color="black").encode(
    x="date:T",
    y=alt.value(100),  # position near top
    text="note:N"
)

# Combine
final_chart = (line_chart + annotation_layer + text_layer).interactive()
st.altair_chart(final_chart, use_container_width=True)








# Define shape mapping based on Type (you can customize this mapping)
shape_map = {
    "multimodal": "circle",
    "chat models (RLHF, DPO)": "square",
    "fine-tuned on domain": "diamond",
    "pretrained": "triangle",
    "continuously pretrained": "cross",
    "base merges and modifications": "star"
}

# Manually create a condition for point shapes
shape_condition = alt.condition(
    alt.datum.Type,
    alt.Shape('Type:N', scale=alt.Scale(domain=list(shape_map.keys()), range=list(shape_map.values()))),
    alt.value('circle')  # fallback
)

# Scatter: CO₂ vs Average Score with enhancements
scatter = alt.Chart(grouped).mark_point(filled=True).encode(
    x=alt.X("CO₂ cost (kg):Q", title="CO₂ Emissions (kg)"),
    y=alt.Y("Average ⬆️:Q", title="Average Score"),
    color=alt.Color("Type:N", legend=alt.Legend(title="Model Type")),
    shape=shape_condition,
    size=alt.Size("Model Count:Q", legend=alt.Legend(title="Number of Models"), scale=alt.Scale(range=[60, 300])),
    tooltip=[
        alt.Tooltip("Type:N", title="Model Type"),
        alt.Tooltip("Average ⬆️:Q", title="Avg. Score"),
        alt.Tooltip("CO₂ cost (kg):Q", title="CO₂ (kg)"),
        alt.Tooltip("Model Count:Q", title="Count")
    ]
).properties(
    title="Model Type Efficiency: Score vs CO₂",
).interactive()

# Optional: Efficiency frontier (score = constant * emission)
line = alt.Chart(grouped).transform_regression(
    "CO₂ cost (kg)", "Average ⬆️"
).mark_line(strokeDash=[4, 4], color="gray").encode(
    x="CO₂ cost (kg):Q",
    y="Average ⬆️:Q"
)

# Optional: Highlight best model
best_point = alt.Chart(grouped).transform_window(
    rank='rank(Average ⬆️)',
    sort=[alt.SortField("Average ⬆️", order='descending')]
).transform_filter("datum.rank == 1").mark_text(
    align='left',
    dx=5,
    dy=-5,
    fontWeight="bold"
).encode(
    x="CO₂ cost (kg):Q",
    y="Average ⬆️:Q",
    text=alt.value("🏆 Best")
)

# Combine charts
st.altair_chart((scatter + line + best_point), use_container_width=True)

# --- User Satisfaction vs Score ---

satisfaction = alt.Chart(grouped).mark_point(filled=True).encode(
    x=alt.X("Hub ❤️:Q", title="User Satisfaction (Hub ❤️)"),
    y=alt.Y("Average ⬆️:Q", title="Average Score"),
    color=alt.Color("Type:N", legend=alt.Legend(title="Model Type")),
    shape=shape_condition,
    size=alt.Size("Model Count:Q", legend=alt.Legend(title="Number of Models"), scale=alt.Scale(range=[60, 300])),
    tooltip=[
        alt.Tooltip("Type:N", title="Model Type"),
        alt.Tooltip("Hub ❤️:Q", title="User Satisfaction"),
        alt.Tooltip("Average ⬆️:Q", title="Avg. Score"),
        alt.Tooltip("Model Count:Q", title="Count")
    ]
).properties(
    title="User Satisfaction vs Average Score",
).interactive()

st.altair_chart(satisfaction, use_container_width=True)


