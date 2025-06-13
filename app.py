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
bar_count = alt.Chart(grouped).mark_bar().encode(
    x=alt.X("Type:N", sort="-y"),
    y=alt.Y("Model Count:Q"),
    color="Type:N",
    tooltip=["Model Count"]
).properties(title="Number of Models by Type")

st.altair_chart(bar_count, use_container_width=True)

# Raw Data Toggle
with st.expander("🔍 Show Raw Filtered Data"):
    st.dataframe(df.sort_values("Average ⬆️", ascending=False))

