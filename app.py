import pandas as pd
import altair as alt
import streamlit as st

# Load and prepare the data
df = pd.read_csv("open-llm-leaderboards.csv")

# Define the aggregate column name
avg_BBH_col = 'Average BBH'

# Group by 'Type' and calculate the median of 'BBH'
df_agg = df.groupby('Type')['BBH'].mean().reset_index(name=avg_BBH_col)

# Create the chart with color encoding by Type
chart = alt.Chart(df_agg).mark_bar().encode(
    x=alt.X('Type:N', sort='-y', title='Type'),
    y=alt.Y(f'{avg_BBH_col}:Q', title='Mean BBH'),
    color=alt.Color('Type:N', title='Type'),  # This line adds different colors for each bar
    tooltip=['Type', avg_BBH_col]
).properties(
    width=700,
    height=400,
    title='Type vs BBH'
).configure_axisX(
    labelAngle=-45
)

chart



