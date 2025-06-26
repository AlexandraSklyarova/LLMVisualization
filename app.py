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
score_cols = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR']

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

st.title("Performance, Pollution, and Popularity: An Exploration of Open LLM Leaderboards")


st.markdown("""
This dashboard explores the function, enjoyment, and environmental impact of open large language models (LLMs) submitted to the Hugging Face leaderboard.  
Users can interactively compare model types across evaluation metrics, track model releases over time, and examine trade-offs between utility and cost using the 
<span style='color:red'>SIDEBAR FILTER</span>.
""", unsafe_allow_html=True)

st.markdown(
    "<hr style='height:3px;border:none;color:#333;background-color:#333;'/>",
    unsafe_allow_html=True
)


st.markdown("###  Model Type Key")
st.markdown("Learn about the types of models included in this dashboard:")

model_type_explanations = {
    "🟢 Pretrained Model": """
**Pretrained Model**
- Newly trained from scratch using masked or causal language modeling.
- Trained on large corpora like Common Crawl, Wikipedia, books, etc.
- Forms the backbone of more advanced fine-tuned or merged models.
""",
    "🟩 Continuously Pretrained Model": """
**Continuously Pretrained Model**
- Built from existing pretrained models.
- Further trained on more recent or curated corpora.
- May include instruction tuning, domain-specific datasets, or chat data.
""",
    "🔶 Fine-Tuned on Domain-Specific Datasets": """
**Fine-Tuned Model**
- Pretrained models adapted to specific domains (e.g., medical, legal).
- Fine-tuned on additional data without modifying architecture.
- Usually boosts performance on target tasks at the cost of generality.
""",
    "💬 Chat Models (RLHF, DPO, IFT…)": """
**Chat Models**
- Trained using **Instruction-Following Tuning (IFT)**, **Reinforcement Learning with Human Feedback (RLHF)**, or **Direct Preference Optimization (DPO)**.
- Tailored for natural conversation, task following, and user preference alignment.
- Can output safer, more controllable text in interactive settings.
""",
    "🤝 Base Merges and MoErges": """
**Merged Models (Base Merges / MoErges)**
- Created by combining weights from multiple base or fine-tuned models.
- May use **merge techniques** like addition, interpolation, or LoRA stacking.
- Usually *not* trained further after merging.
- Useful for combining capabilities or styles.
""",
    "🌸 Multimodal Models": """
**Multimodal Models**
- Can process inputs from different modalities like **text**, **images**, **audio**, or **video**.
- Examples include image captioning, vision-language models, and audio reasoning.
- Represent the next frontier in general AI capabilities.
"""
}

# Display with expanders
for label, description in model_type_explanations.items():
    with st.expander(label):
        st.markdown(description)


st.markdown(
    "<hr style='height:3px;border:none;color:#333;background-color:#333;'/>",
    unsafe_allow_html=True
)



st.header("Evaluation of Different LLM Models")

st.markdown("""
In this section, you can explore the abilities of different LLM types based on various metrics (explained below). 
Use the <span style='color:red'>LEGEND TO FILTER</span> by type of model. 
Note that the scores are normalized and range from 0–100.
""", unsafe_allow_html=True)


#new

long_df = grouped.melt(
    id_vars=["Type"],
    value_vars=score_cols,
    var_name="Metric",
    value_name="Score"
)

legend_selection = alt.selection_point(fields=["Type"], bind="legend")

# ---- BASE BAR ----
base = alt.Chart(long_df).mark_bar().encode(
    x=alt.X("Type:N", title="Model Type"),
    y=alt.Y("Score:Q", title="Average Score",
            scale=alt.Scale(domain=[0, 100])),  # ✅ Fixed range
    color=alt.Color("Type:N", legend=alt.Legend(title="Model Type")),
    opacity=alt.condition(legend_selection, alt.value(1.0), alt.value(0.2)),
    tooltip=[
        alt.Tooltip("Type:N", title="Model Type"),
        alt.Tooltip("Metric:N", title="Evaluation Metric"),
        alt.Tooltip("Score:Q", format=".2f")
    ]
).add_params(legend_selection)

# ---- LABELS ----
labels = alt.Chart(long_df).mark_text(
    align="center",
    baseline="bottom",
    dy=-5,
    fontSize=9
).encode(
    x="Type:N",
    y="Score:Q",
    text=alt.Text("Score:Q", format=".2f"),
    opacity=alt.condition(legend_selection, alt.value(1.0), alt.value(0.2))
)

# ---- COMBINE + FACET ----
chart = (base + labels).facet(
    column=alt.Column("Metric:N", title=None, header=alt.Header(labelAngle=0))
).properties(
    title="Scores by Model Type across Evaluation Metrics",
    spacing=30
).resolve_scale(
    y="shared"  # Single, flexible y-axis across metrics
)

# ---- DISPLAY ----
st.altair_chart(chart, use_container_width=True)






evaluation_summary = {
    "IFEval": {
        "Description": """
**IFEval** checks whether a model can follow explicit formatting instructions.

- Focuses on format adherence, not correctness  
- Includes tasks like:  
  - *“Include the keyword X”*  
  - *“Respond using format Y”*  
- Useful for evaluating models as tools in pipeline-style workflows
"""
    },
    "BBH": {
        "Description": """
**BBH** (Beyond the Imitation Game Benchmark) is a challenging reasoning benchmark.

- Covers 23 BigBench tasks including:  
  - Logic puzzles  
  - Math word problems  
  - Symbolic reasoning  
- Highly correlated with human preference judgments
"""
    },
    "MATH Lvl 5": {
        "Description": """
**MATH Lvl 5** consists of high school math competition problems.

- Includes problems from contests like AMC and AIME  
- Requires exact formatting (LaTeX, Asymptote)  
- Emphasizes correctness, clarity, and structured output
"""
    },
    "GPQA": {
        "Description": """
**GPQA** is a graduate-level STEM question set vetted by domain experts.

- Topics include:  
  - Biology  
  - Chemistry  
  - Physics  
- Designed to avoid test-set contamination  
- Emphasizes true conceptual depth over pattern matching
"""
    },
    "MuSR": {
        "Description": """
**MuSR** (Multistep Structured Reasoning) features long-context challenges.

- Includes:  
  - Logic puzzles  
  - Mystery reasoning  
  - Logistics chains  
- Tests the model's ability to track dependencies over long input
"""
    }
}


st.markdown("### Evaluation Metric Descriptions")

for eval_name, content in evaluation_summary.items():
    with st.expander(eval_name):
        st.markdown(content["Description"])




st.markdown(
    "<hr style='height:3px;border:none;color:#333;background-color:#333;'/>",
    unsafe_allow_html=True
)


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
event_date = pd.to_datetime("2024-05-13")
event_df = pd.DataFrame({
    "date": [event_date],
    "label": ["Release of GPT-4o"]
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
    dy=5,  # Slight positive shift downward for margin
    fontSize=11,
    fontStyle="italic",
    color="red"
).encode(
    x="date:T",
    y=alt.value(10),  # Fixed y-value ensures the label stays in view
    text="label:N"
)

# --- Combine everything ---
final_chart = alt.layer(line_chart, event_rule, event_text).properties(
    title="Cumulative Number of LLM Models Released Over Time (Zoom Enabled)",
    width=1400,
    height=600
)

st.altair_chart(final_chart, use_container_width=True)







st.markdown(
    "<hr style='height:3px;border:none;color:#333;background-color:#333;'/>",
    unsafe_allow_html=True
)




st.header("CO₂ Emissions Overview")

st.markdown("""
Here’s a breakdown of average CO₂ emissions per model type. Use the <span style='color:red'>LEGEND TO FILTER</span>.
""", unsafe_allow_html=True)

with st.expander("How Hugging Face LLM Leaderboard Collects CO₂ Data"):
    st.markdown("""
    - The LLM Leaderboard **doesn't measure CO₂ emissions itself**.  
    - It **displays CO₂ estimates** sourced from:
        - Model cards uploaded by the creators.
        - The **ML CO₂ Impact Project** and similar efforts.
    - These estimates typically come from tools like:
        - [CodeCarbon](https://codecarbon.io/)
        - [Experiment Impact Tracker](https://www.impact-tracker.org/)
    - Not every model has CO₂ data. It's only available if the creator includes it.
    """)



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
    size=alt.Size("Size:Q", scale=alt.Scale(range=[300, 20000]), legend=None),
    color=alt.Color("Type:N", legend=alt.Legend(title="Model Type")),
    opacity=alt.condition(type_selection, alt.value(1.0), alt.value(0.2)),
    tooltip=["Type:N", "CO₂ cost (kg):Q"]
).add_params(type_selection).properties(
    title="Average CO₂ Output per Model Type (Radius = CO₂ Cost (kg))",
    width=500,
    height=300
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



st.markdown(
    "<hr style='height:3px;border:none;color:#333;background-color:#333;'/>",
    unsafe_allow_html=True
)



st.header("Relationship between User Likes and Average Scores of LLMs")

st.markdown("""
Here you can explore how popular models are in relation to how they perform across different metrics. 
Use the <span style='color:red'>BRUSH FEATURE IN THE LEFT CHART TO FILTER</span> both charts. 
Filtering by time in the sidebar can reveal even more about their relationship.
""", unsafe_allow_html=True)







df.columns = df.columns.str.strip()
df = df.rename(columns={"Average ⬆️": "Average"})
df["Hub ❤️"] = pd.to_numeric(df["Hub ❤️"], errors="coerce")
df["Average"] = pd.to_numeric(df["Average"], errors="coerce")
df = df.dropna(subset=["Average", "Hub ❤️", "eval_name"])
df["Average_Bin"] = ((df["Average"] // 5) * 5).astype(int)

# ---- AGGREGATION ----
binned_avg = df.groupby("Average_Bin", as_index=False).agg(
    Mean_Hub_Score=("Hub ❤️", "mean"),
    Eval_Count=("eval_name", "count")
)

# ---- BRUSH SELECTION ----
brush = alt.selection_interval(encodings=["x"])

# ---- LEFT CHART ----
heatmap = alt.Chart(binned_avg).mark_bar().encode(
    x=alt.X("Average_Bin:O", title="Average Score Bin (5 pt range)"),
    y=alt.Y("Mean_Hub_Score:Q",
            title="Mean Number of Likes"
    ),  # 👈 CLOSE the Y channel
    color=alt.condition(
        brush,
        alt.Color("Mean_Hub_Score:Q",
                  scale=alt.Scale(scheme="goldred"),
                  title="Mean Likes"),
        alt.value("lightgray")
    ),
    tooltip=[
        alt.Tooltip("Average_Bin:O", title="Average Score Bin"),
        alt.Tooltip("Mean_Hub_Score:Q", title="Mean Likes", format=".1f"),
        alt.Tooltip("Eval_Count:Q", title="Number of Models")
    ]
).add_params(brush).properties(
    title="Distribution of Likes by Average Score",
    width=300,
    height=400
)

# ---- RIGHT CHART ----
points = alt.Chart(df).mark_circle(size=40, opacity=0.5).encode(
    x=alt.X("Average:Q", title="Average Score"),
    y=alt.Y("Hub ❤️:Q", title="Hub Likes"),
    color=alt.Color("Hub ❤️:Q", scale=alt.Scale(scheme="goldred")),
    tooltip=[
        alt.Tooltip("eval_name:N", title="Model Name"),
        alt.Tooltip("Average:Q", title="Average Score", format=".1f"),
        alt.Tooltip("Hub ❤️:Q", title="Hub Likes", format=".1f")
    ]
).transform_filter(brush).properties(
    title="Models in Selected Score Bin",
    width=300,
    height=400
)

# ---- DISPLAY BOTH ----
st.altair_chart(heatmap | points, use_container_width=True)

st.markdown("""
###  Key Conclusions
-  **Rapid Growth:** The number of LLMs has surged, especially after mid‑2024, reflecting the accelerating pace of AI development.  
-  **Top Performers:** 
   - Models generally score highest on **IFEval** and lowest on **GPQA**, suggesting certain benchmarks remain more challenging. 
   - Multimodal models did the best on average, however, they are also the newest so the data could be skewed in their favor.
-  **Common Model Types:** Fine‑tuned models and base merges make up the bulk of available LLMs.  
-  **CO₂ Impact:**  
    - Multimodal models have the highest average CO₂ output per model.  
    - Fine‑tuned models have contributed the most to total CO₂ emissions due to their sheer quantity.  
-  **Performance vs. User Satisfaction:** Higher average scores don’t necessarily imply higher user satisfaction unless you control for the timeframe and context.  
-  **Patterns over Time:** Model quality and popularity evolve rapidly. Filtering by date can reveal important shifts and trends.  
-  **Implication for Users:** Evaluating LLMs benefits from considering both performance metrics and environmental impacts for a more holistic view.
""", unsafe_allow_html=True)
