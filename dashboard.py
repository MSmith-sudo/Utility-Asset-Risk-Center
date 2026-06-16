import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #111827;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

h1 {
    color: #F8FAFC;
}

h2, h3 {
    color: #CBD5E1;
}

div[data-testid="stMetric"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)

DATA_PATH = "data/infrastructure_failures.csv"

RISK_LABELS = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk"
}

RISK_ACTIONS = {
    0: "Continue standard monitoring.",
    1: "Schedule preventive maintenance and increase monitoring frequency.",
    2: "Immediate inspection recommended. Consider replacement planning."
}


st.set_page_config(
    page_title="Utility Asset Risk Center",
    page_icon="🚰",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
    }
    div[data-testid="stMetric"] {
        background-color: #111827;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #334155;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚰 Utility Asset Risk Center")
st.caption("Predictive maintenance and infrastructure failure analytics for water utility assets.")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(df):
    df_encoded = pd.get_dummies(
        df,
        columns=["material", "soil_type", "location_type"],
        drop_first=False
    )

    X = df_encoded.drop(["pipe_id", "failure_risk"], axis=1)
    y = df_encoded["failure_risk"]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X, y)

    return model, X.columns


def calculate_asset_health(age, previous_breaks, leak_reports, pressure):
    score = 100
    score -= age * 0.45
    score -= previous_breaks * 5
    score -= leak_reports * 3
    score -= max(0, pressure - 90) * 0.15
    return max(0, min(100, score))


def get_priority_label(prediction, health_score):
    if prediction == 2 or health_score < 35:
        return "Critical Priority"
    elif prediction == 1 or health_score < 65:
        return "Monitor Closely"
    return "Routine Monitoring"


def prepare_scenario(
    age,
    material,
    diameter,
    pressure,
    soil_type,
    previous_breaks,
    location_type,
    leak_reports,
    model_columns
):
    scenario = pd.DataFrame([{
        "age": age,
        "diameter": diameter,
        "pressure": pressure,
        "previous_breaks": previous_breaks,
        "leak_reports": leak_reports,
        f"material_{material}": 1,
        f"soil_type_{soil_type}": 1,
        f"location_type_{location_type}": 1
    }])

    for col in model_columns:
        if col not in scenario.columns:
            scenario[col] = 0

    return scenario[model_columns]


def percent_difference(value, average):
    if average == 0:
        return 0
    return ((value - average) / average) * 100


def explain_prediction(df, scenario_values, prediction, model, model_columns):
    age = scenario_values["age"]
    pressure = scenario_values["pressure"]
    previous_breaks = scenario_values["previous_breaks"]
    leak_reports = scenario_values["leak_reports"]
    diameter = scenario_values["diameter"]

    health_score = calculate_asset_health(
        age,
        previous_breaks,
        leak_reports,
        pressure
    )

    priority = get_priority_label(prediction, health_score)

    avg_age = df["age"].mean()
    avg_pressure = df["pressure"].mean()
    avg_breaks = df["previous_breaks"].mean()
    avg_leaks = df["leak_reports"].mean()
    avg_diameter = df["diameter"].mean()

    age_diff = percent_difference(age, avg_age)
    pressure_diff = percent_difference(pressure, avg_pressure)
    breaks_diff = percent_difference(previous_breaks, avg_breaks)
    leaks_diff = percent_difference(leak_reports, avg_leaks)
    diameter_diff = percent_difference(diameter, avg_diameter)

    importance_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    readable_importance = importance_df.copy()
    readable_importance["Feature"] = (
        readable_importance["Feature"]
        .str.replace("_", " ")
        .str.title()
    )

    top_feature = readable_importance.iloc[0]["Feature"]

    st.divider()
    st.subheader("Asset Risk Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 2:
            st.error(f"Failure Risk: {RISK_LABELS[prediction]}")
        elif prediction == 1:
            st.warning(f"Failure Risk: {RISK_LABELS[prediction]}")
        else:
            st.success(f"Failure Risk: {RISK_LABELS[prediction]}")

    with col2:
        st.metric("Asset Health Score", f"{health_score:.0f}/100")

    with col3:
        st.metric("Inspection Priority", priority)

    st.subheader("Maintenance Recommendation")
    st.info(RISK_ACTIONS[prediction])

    st.subheader("Why this prediction?")

    reasons = [
        f"Pipe age is {abs(age_diff):.1f}% {'above' if age_diff >= 0 else 'below'} the dataset average.",
        f"Pressure is {abs(pressure_diff):.1f}% {'above' if pressure_diff >= 0 else 'below'} the dataset average.",
        f"Previous breaks are {abs(breaks_diff):.1f}% {'above' if breaks_diff >= 0 else 'below'} the dataset average.",
        f"Leak reports are {abs(leaks_diff):.1f}% {'above' if leaks_diff >= 0 else 'below'} the dataset average.",
        f"Pipe diameter is {abs(diameter_diff):.1f}% {'above' if diameter_diff >= 0 else 'below'} the dataset average.",
        f"The model identified {top_feature} as the strongest overall risk driver."
    ]

    for reason in reasons:
        st.write(f"- {reason}")

    st.info(
        "Prototype note: this project uses synthetic infrastructure data and simulated risk labels for portfolio demonstration."
    )

    st.subheader("Asset Comparison Against Portfolio Average")

    comparison_df = pd.DataFrame({
        "Metric": ["Age", "Diameter", "Pressure", "Previous Breaks", "Leak Reports"],
        "Scenario Value": [age, diameter, pressure, previous_breaks, leak_reports],
        "Portfolio Average": [avg_age, avg_diameter, avg_pressure, avg_breaks, avg_leaks],
        "Difference (%)": [age_diff, diameter_diff, pressure_diff, breaks_diff, leaks_diff]
    })

    st.dataframe(comparison_df, use_container_width=True)

    st.subheader("Top Risk Drivers")

    st.bar_chart(readable_importance.head(10).set_index("Feature"))
    st.dataframe(readable_importance, use_container_width=True)


df = load_data()
model, model_columns = train_model(df)

st.sidebar.title("Operations Filters")

selected_materials = st.sidebar.multiselect(
    "Filter by material",
    sorted(df["material"].unique()),
    default=sorted(df["material"].unique())
)

selected_locations = st.sidebar.multiselect(
    "Filter by location type",
    sorted(df["location_type"].unique()),
    default=sorted(df["location_type"].unique())
)

filtered_df = df[
    df["material"].isin(selected_materials)
    & df["location_type"].isin(selected_locations)
]

st.sidebar.info(
    """
    This dashboard is designed like a utility operations tool:
    portfolio view, risk assessment, maintenance planning, and data exploration.
    """
)

st.subheader("Utility Portfolio Snapshot")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Assets in View", len(filtered_df))

with col2:
    st.metric("High-Risk Assets", len(filtered_df[filtered_df["failure_risk"] == 2]))

with col3:
    st.metric("Avg Pipe Age", f"{filtered_df['age'].mean():.1f} yrs")

with col4:
    st.metric("Avg Pressure", f"{filtered_df['pressure'].mean():.1f} psi")


portfolio_tab, assessment_tab, maintenance_tab, data_tab = st.tabs([
    "Asset Portfolio",
    "Risk Assessment",
    "Maintenance Planning",
    "Data Explorer"
])


with portfolio_tab:
    st.subheader("Portfolio Risk Breakdown")

    risk_counts = filtered_df["failure_risk"].map(RISK_LABELS).value_counts()
    st.bar_chart(risk_counts)

    st.subheader("Asset Mix by Material")

    material_counts = filtered_df["material"].value_counts()
    st.bar_chart(material_counts)

    st.subheader("Average Infrastructure Conditions")

    avg_df = pd.DataFrame({
        "Metric": ["Age", "Diameter", "Pressure", "Previous Breaks", "Leak Reports"],
        "Average": [
            filtered_df["age"].mean(),
            filtered_df["diameter"].mean(),
            filtered_df["pressure"].mean(),
            filtered_df["previous_breaks"].mean(),
            filtered_df["leak_reports"].mean()
        ]
    })

    st.dataframe(avg_df, use_container_width=True)


with assessment_tab:
    st.subheader("Custom Pipe Failure Risk Assessment")

    left, right = st.columns(2)

    with left:
        age = st.slider("Pipe Age", 1, 100, 45)
        diameter = st.slider("Pipe Diameter", 4, 48, 12)
        pressure = st.slider("Water Pressure", 20, 150, 80)
        previous_breaks = st.slider("Previous Breaks", 0, 10, 2)
        leak_reports = st.slider("Leak Reports", 0, 15, 3)

    with right:
        material = st.selectbox("Pipe Material", sorted(df["material"].unique()))
        soil_type = st.selectbox("Soil Type", sorted(df["soil_type"].unique()))
        location_type = st.selectbox("Location Type", sorted(df["location_type"].unique()))

        preview_health = calculate_asset_health(
            age,
            previous_breaks,
            leak_reports,
            pressure
        )

        st.metric("Live Asset Health Estimate", f"{preview_health:.0f}/100")

    if st.button("Run Risk Assessment"):
        scenario = prepare_scenario(
            age=age,
            material=material,
            diameter=diameter,
            pressure=pressure,
            soil_type=soil_type,
            previous_breaks=previous_breaks,
            location_type=location_type,
            leak_reports=leak_reports,
            model_columns=model_columns
        )

        prediction = model.predict(scenario)[0]

        scenario_values = {
            "age": age,
            "diameter": diameter,
            "pressure": pressure,
            "previous_breaks": previous_breaks,
            "leak_reports": leak_reports
        }

        explain_prediction(
            df=df,
            scenario_values=scenario_values,
            prediction=prediction,
            model=model,
            model_columns=model_columns
        )


with maintenance_tab:
    st.subheader("Maintenance Planning Queue")

    high_risk_assets = filtered_df[filtered_df["failure_risk"] == 2].copy()
    high_risk_assets = high_risk_assets.sort_values(
        by=["previous_breaks", "leak_reports", "age"],
        ascending=False
    )

    st.write("Highest-priority assets based on failure risk, break history, leak reports, and age.")

    st.dataframe(
        high_risk_assets.head(25),
        use_container_width=True
    )

    st.subheader("Recommended Next Steps")

    st.markdown(
        """
        - Inspect high-risk pipes with repeated break history.
        - Prioritize older cast iron and steel assets.
        - Increase monitoring for pipes with frequent leak reports.
        - Use this model as a screening tool, not as a replacement for engineering judgment.
        """
    )


with data_tab:
    st.subheader("Infrastructure Dataset Explorer")

    st.dataframe(filtered_df, use_container_width=True)

    csv = filtered_df.to_csv(index=False)

    st.download_button(
        label="Download Filtered Dataset",
        data=csv,
        file_name="filtered_infrastructure_failures.csv",
        mime="text/csv"
    )