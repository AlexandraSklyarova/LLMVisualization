import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

st.set_page_config(page_title="LLM Leaderboard Dashboard", layout="wide")

# Load data
df = pd.read_csv("open-llm-leaderboards.csv")
df.columns = df.columns.str.strip()

# Show available columns
st.sidebar.write("🧠 Columns:", df.columns.tolist())

# Helper to find column
def find_column(possible_names):
    for name in possible_names:
        for col in df.columns:
            if col.lower().replace(" ", "_") == name.lower().replace(" ", "_"):
                return col
    return None

# Auto-detect columns
model_col = find_column(["model", "model_name", "Model Name"])
date_col = find_column(["submission_date", "date", "created_at"])
co2_col = find_column(["carbon_footprint", "co2", "carbon"])
user_score_col = find_column(["user_score", "user rating", "satisfaction"])
score_cols = [find_column([s]) for s in ["IFEval", "BBH", "MUSR", "MATH LvL5", "GPQA"]]
score_cols = [col for col in score_cols if col in df.columns and df[col].dtype in [np.float64, np.int64]]

# Add average score
if score_cols:
    df["Average Score"] = df[score_cols].mean(axis=1)

# Sidebar model filter
if model_col:
    models = df[model_col].dropna().unique().tolist()
    selected_models = st.sidebar.multiselect("Select models:", models, default=models[:5])
    df = df[df[model_col].isin(selected_models)]

# Date filtering (safe version)
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    valid_dates = df[date_col].dropna()
    if not valid_dates.empty:
        min_date, max_date = valid_dates.min(), valid_dates.max()
        date_range = st.sidebar.slider("Submission Date Range:", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        df = df[(df[date_col] >= date_range[0]) & (df[date_col] <= date_range[1])]

# --- Visualizations ---
st.title("🤖 Open LLM Leaderboard Dashboard")

# 1. Radar Chart
if model_col and len(score_cols) >= 3:
    st.subheader("📊 Model Scores (Radar)")
    radar_data = df[[model_col] + score_cols].dropna()
    radar_melt = radar_data.melt(id_vars=model_col, var_name="Metric", value_name="Score")
    radar_chart = alt.Chart(radar_melt).mark_line(point=True).encode(
        theta=alt.Theta("Metric:N", sort=score_cols),
        radius=alt.Radius("Score:Q", scale=alt.Scale(type="linear")),
        color=f"{model_col}:N"
    ).properties(height=400)
    st.altair_chart(radar_chart, use_container_width=True)

# 2. CO2 vs Avg Score
if co2_col and co2_col in df.columns:
    st.subheader("🌍 CO₂ vs. Average Score")
    scatter = alt.Chart(df).mark_circle(size=80).encode(
        x=f"{co2_col}:Q",
        y="Average Score:Q",
        color=f"{model_col}:N",
        tooltip=[model_col, co2_col, "Average Score"]
    ).interactive()
    st.altair_chart(scatter, use_container_width=True)

# 3. Scores Over Time
if date_col and not df[date_col].isna().all():
    st.subheader("📈 Performance Over Time")
    time_chart = alt.Chart(df).mark_line(point=True).encode(
        x=f"{date_col}:T",
        y="Average Score:Q",
        color=f"{model_col}:N",
        tooltip=[model_col, "Average Score"]
    ).interactive()
    st.altair_chart(time_chart, use_container_width=True)

# 4. Score Correlation
if len(score_cols) >= 2:
    st.subheader("📐 Score Correlation")
    corr = df[score_cols].corr()
    corr_df = corr.stack().reset_index()
    corr_df.columns = ['Metric1', 'Metric2', 'Correlation']
    heatmap = alt.Chart(corr_df).mark_rect().encode(
        x="Metric1:O",
        y="Metric2:O",
        color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="purplegreen")),
        tooltip=["Metric1", "Metric2", "Correlation"]
    )
    st.altair_chart(heatmap, use_container_width=True)

# 5. User Satisfaction Histogram
if user_score_col and user_score_col in df.columns:
    st.subheader("👍 User Satisfaction")
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{user_score_col}:Q", bin=True),
        y="count()",
        color=f"{model_col}:N",
        tooltip=["count()"]
    ).interactive()
    st.altair_chart(hist, use_container_width=True)

# 6. Data Table
with st.expander("🔍 View Filtered Data"):
    st.dataframe(df.reset_index(drop=True))
