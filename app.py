import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

st.set_page_config(page_title="LLM Leaderboard Dashboard", layout="wide")

# Load the data
df = pd.read_csv("open-llm-leaderboards.csv")

# Show column names for debugging
st.sidebar.write("📌 Available columns:", df.columns.tolist())

# Optional: Parse date column if it exists
if "submission_date" in df.columns:
    df["submission_date"] = pd.to_datetime(df["submission_date"], errors='coerce')

st.title("🤖 Open LLM Leaderboard Dashboard")
st.markdown("Explore performance, environmental impact, and user ratings of top LLMs.")

# Sidebar filters
model_options = df["model_name"].unique().tolist()
selected_models = st.sidebar.multiselect("Select models to compare:", model_options, default=model_options[:6])

filtered_df = df[df["model_name"].isin(selected_models)]

# Optional date filter
if "submission_date" in df.columns:
    min_date = filtered_df["submission_date"].min()
    max_date = filtered_df["submission_date"].max()
    date_range = st.sidebar.slider("Filter by submission date:", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    filtered_df = filtered_df[
        (filtered_df["submission_date"] >= date_range[0]) &
        (filtered_df["submission_date"] <= date_range[1])
    ]

# Define score metrics
score_cols = ["IFEval", "BBH", "MUSR", "MATH LvL5", "GPQA"]
valid_scores = [col for col in score_cols if col in df.columns]

# 1. Radar Chart of Scores
if len(valid_scores) >= 3:
    st.subheader("📊 Model Performance Comparison (Radar Plot)")
    radar_data = filtered_df[["model_name"] + valid_scores].dropna()

    radar_melt = radar_data.melt(id_vars=["model_name"], var_name="Metric", value_name="Score")

    radar_chart = alt.Chart(radar_melt).mark_line(point=True).encode(
        theta=alt.Theta("Metric:N", sort=valid_scores),
        radius=alt.Radius("Score:Q", scale=alt.Scale(type="linear", zero=True)),
        color="model_name:N"
    ).properties(height=400, width=400)

    st.altair_chart(radar_chart, use_container_width=True)

# 2. CO₂ Emissions vs Score
if "carbon_footprint" in df.columns and "Average Score" not in df.columns:
    df["Average Score"] = df[valid_scores].mean(axis=1)

if "carbon_footprint" in df.columns:
    st.subheader("🌍 Environmental Impact vs Score")
    scatter = alt.Chart(filtered_df).mark_circle(size=100).encode(
        x="carbon_footprint:Q",
        y="Average Score:Q",
        tooltip=["model_name", "carbon_footprint", "Average Score"],
        color="model_name:N"
    ).interactive().properties(height=400)

    st.altair_chart(scatter, use_container_width=True)

# 3. Time Series of Score (if date column exists)
if "submission_date" in df.columns:
    st.subheader("📈 Model Performance Over Time")
    time_data = filtered_df.copy()
    time_data["Average Score"] = time_data[valid_scores].mean(axis=1)

    line = alt.Chart(time_data).mark_line(point=True).encode(
        x="submission_date:T",
        y="Average Score:Q",
        color="model_name:N",
        tooltip=["model_name", "submission_date", "Average Score"]
    ).interactive().properties(height=400)

    st.altair_chart(line, use_container_width=True)

# 4. Correlation Heatmap
if len(valid_scores) >= 2:
    st.subheader("🔗 Score Correlation Heatmap")
    corr_data = df[valid_scores].corr()
    corr_df = corr_data.stack().reset_index()
    corr_df.columns = ['Metric1', 'Metric2', 'Correlation']

    heatmap = alt.Chart(corr_df).mark_rect().encode(
        x="Metric1:O",
        y="Metric2:O",
        color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="purplebluegreen")),
        tooltip=["Metric1", "Metric2", "Correlation"]
    ).properties(height=400)

    st.altair_chart(heatmap, use_container_width=True)

# 5. User Satisfaction Distribution
if "user_score" in df.columns:
    st.subheader("🧑‍💻 User Satisfaction Distribution")
    hist = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X("user_score:Q", bin=alt.Bin(maxbins=20)),
        y="count()",
        color="model_name:N",
        tooltip=["count()"]
    ).interactive().properties(height=300)

    st.altair_chart(hist, use_container_width=True)

# 6. Raw Data Table
with st.expander("🔍 Explore Raw Data"):
    st.dataframe(filtered_df.reset_index(drop=True))
