import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Tornado Research Project",
    page_icon="🌪️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("/content/demo_model.joblib")

model = load_model()

# Sidebar
st.sidebar.title("🌪️ Tornado Research")
st.sidebar.write("AI-Based Weather Analysis")
st.sidebar.divider()
st.sidebar.write("**Project status:** Educational prototype")
st.sidebar.write("**Dataset:** 7 simulated records")
st.sidebar.write("**Algorithm:** Decision Tree")

# Main page
st.title("AI-Based Tornado Risk Prediction")
st.caption("College Research Project | Python • Machine Learning • Streamlit")

st.warning(
    "EDUCATIONAL DEMO ONLY. This model uses simulated weather "
    "records and arbitrary labels. Its output is not a real "
    "tornado-risk estimate or an official weather warning."
)

overview, predictor, data_info = st.tabs(
    ["Project Overview", "Weather Demo", "Dataset & Limitations"]
)

with overview:
    st.subheader("Project Overview")
    st.write(
        "This project demonstrates how a machine-learning model "
        "can accept weather measurements and return a model output."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Demo Records", "7")
    col2.metric("Input Features", "4")
    col3.metric("Model", "Decision Tree")

    st.subheader("Weather features")
    st.write(
        "Temperature, relative humidity, wind speed, and air pressure."
    )

with predictor:
    st.subheader("Enter example weather measurements")
    st.write("Change the values and run the educational model.")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-50.0, max_value=60.0,
            value=30.0, step=1.0
        )
        humidity = st.slider(
            "Relative humidity (%)",
            min_value=0, max_value=100, value=70
        )

    with col2:
        wind_speed = st.number_input(
            "Wind speed (km/h)",
            min_value=0.0, max_value=500.0,
            value=25.0, step=1.0
        )
        pressure = st.number_input(
            "Air pressure (hPa)",
            min_value=1.0, max_value=1200.0,
            value=1005.0, step=1.0
        )

    if st.button("Run Demo Prediction", type="primary"):
        weather = pd.DataFrame([{
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "wind_speed_kmh": wind_speed,
            "pressure_hpa": pressure
        }])

        prediction = int(model.predict(weather)[0])

        st.subheader("Model output")
        st.info(f"Demonstration label: {prediction}")
        st.caption(
            "This is an arbitrary label from a model trained on "
            "simulated data. It is not a probability, a verified "
            "risk level, or a tornado warning."
        )

        st.subheader("Your entered weather values")
        st.dataframe(weather, use_container_width=True)

with data_info:
    st.subheader("Dataset information")
    st.write(
        "The current dataset contains seven simulated examples "
        "created to test the software workflow."
    )

    st.subheader("Important limitations")
    st.markdown(
        """
        - The records are not verified historical observations.
        - The model labels are arbitrary demonstration labels.
        - Training accuracy does not establish forecasting ability.
        - No real-world tornado prediction has been validated.
        - Real research requires suitable weather observations,
          verified event records, and independent evaluation.
        """
    )

st.divider()
st.caption(
    "Educational research prototype • Not for operational "
    "weather forecasting or safety decisions"
)
