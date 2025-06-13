import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

# --- Load Data ---
df = pd.read_csv("open-llm-leaderboards.csv")

# --- Preprocess ---
df["submission_date"] = pd.to_datetime(df["submission_date"], errors='coerce')
df["average_score"] = df[["IFEval_score", "BBH_score", "MUSR_score", "MATH_level_5_score", "GPQA_score"]].mean(axis=1)

# --- Sidebar Filters ---
st.sidebar.header("Filters")
model_types = st.sidebar.multiselect("Select Model Type", options=df.model_type.unique(), default=df.model_type.unique())
architectures = st.sidebar.multiselect("Select Architecture", options=df.architecture.unique(), default=df.architecture.unique())
moe = st.sidebar.radio("Mixture of Experts (MoE)?", ["All", True, False], index=0)
date_range = st.sidebar.date_input("Submission Date Range", [df.submission_date.min(), df.submission_date.max()])

# Apply filters
filtered = df[
    df.model_type.isin(model_types) &
    df.architecture.isin(architectures) &
    (df.submission_date >= pd.to_datetime(date_range[0])) &
    (df.submission_date <= pd.to_datetime(date_range[1]))
]
if moe != "All":
    filtered = filtered[filtered.moe == moe]

# --- Layout ---
st.title("📊 Open LLM Leaderboard Dashboard")
st.markdown("Explore model performance, environmental impact, and user engagement.")

# --- Score Comparison Radar Chart ---
st.subheader("Radar Chart: Score Comparison")
selected_models = st.multiselect("Select Models", options=filtered.model.unique(), default=filtered.model.head(3))
radar_data = filtered[filtered.model.isin(selected_models)]
radar = radar_data.melt(id_vars=["model"], value_vars=["IFEval_score", "BBH_score", "MUSR_score", "MATH_level_5_score", "GPQA_score"], 
                         var_name="Score Type", value_name="Score")
radar_chart = alt.Chart(radar).mark_line(point=True).encode(
    theta=alt.Theta("Score Type:N", sort=None),
    radius=alt.Radius("Score:Q", scale=alt.Scale(type='linear', zero=True)),
    color="model:N"
).properties(height=400)
st.altair_chart(radar_chart, use_container_width=True)

# --- Timeline of Average Score ---
st.subheader("Timeline: Average Score Over Time")
time_data = filtered.groupby("submission_date")["average_score"].mean().reset_index()
time_chart = alt.Chart(time_data).mark_line().encode(
    x="submission_date:T",
    y="average_score:Q"
).properties(height=300)
st.altair_chart(time_chart, use_container_width=True)

# --- CO₂ vs. Average Score ---
st.subheader("CO₂ Emissions vs. Average Score")
scatter = alt.Chart(filtered).mark_circle(size=100, opacity=0.6).encode(
    x="carbon_cost_kg:Q",
    y="average_score:Q",
    size="params_b:Q",
    color="model_type:N",
    tooltip=["model", "carbon_cost_kg", "average_score", "params_b"]
).interactive().properties(height=400)
st.altair_chart(scatter, use_container_width=True)

# --- Correlation Heatmap ---
st.subheader("Correlation Heatmap of Scores & Metrics")
metrics = filtered[["IFEval_score", "BBH_score", "MUSR_score", "MATH_level_5_score", "GPQA_score", "carbon_cost_kg", "average_score"]]
corr_df = metrics.corr().reset_index().melt("index")
heatmap = alt.Chart(corr_df).mark_rect().encode(
    x="index:N",
    y="variable:N",
    color="value:Q"
) + alt.Chart(corr_df).mark_text(baseline='middle').encode(
    x="index:N",
    y="variable:N",
    text=alt.Text("value:Q", format=".2f")
)
st.altair_chart(heatmap, use_container_width=True)

# --- User Engagement vs Score ---
st.subheader("User Engagement vs. Average Score")
bar = alt.Chart(filtered).mark_bar().encode(
    x=alt.X("average_score:Q", bin=alt.Bin(maxbins=15)),
    y="count():Q",
    color=alt.Color("❤️_on_HuggingFace:Q", scale=alt.Scale(scheme='redpurple')),
    tooltip=["count()"]
).properties(height=300)
st.altair_chart(bar, use_container_width=True)

# --- Data Table ---
st.subheader("Explore Raw Model Data")
st.dataframe(filtered[["model", "submission_date", "average_score", "carbon_cost_kg", "params_b", "❤️_on_HuggingFace"]].sort_values("average_score", ascending=False))
