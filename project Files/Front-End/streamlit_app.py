# streamlit_app.py - Streamlit Frontend for HealthAI

import streamlit as st
import requests
import json
import os

# --- Configuration ---
# Get the Flask API URL from environment variables.
# This allows flexibility if your Flask app is deployed elsewhere.
# Default to localhost for local development.
FLASK_API_URL = os.getenv("FLASK_API_URL", "http://localhost:5000")

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="HealthAI: Intelligent Healthcare Assistant",
    layout="wide", # Use wide layout for better content display
    initial_sidebar_state="auto"
)

st.title("🩺 HealthAI: Intelligent Healthcare Assistant")

# --- Global Medical Disclaimer ---
st.markdown("""
<div style="background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; border-radius: 5px; padding: 10px; margin-bottom: 20px;">
    <strong>Important Medical Disclaimer:</strong> The information provided by HealthAI is for informational purposes only and does not constitute medical advice, diagnosis, or treatment. It is not a substitute for professional medical advice. Always consult with a qualified healthcare professional for any health concerns, before making any decisions related to your health, or if you have questions about a medical condition. Do not disregard professional medical advice or delay seeking it because of something you have read here.
</div>
""", unsafe_allow_html=True)

# --- Tabbed Interface for Different HealthAI Features ---
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Patient Chat",
    "🔍 Disease Prediction",
    "💊 Treatment Plans",
    "📊 Health Analytics"
])

# --- Function to make POST requests to the Flask backend ---
def call_flask_api(endpoint: str, payload: dict) -> dict:
    """
    Helper function to send POST requests to the Flask backend.
    Handles connection errors and API response parsing.
    """
    try:
        response = requests.post(f"{FLASK_API_URL}/{endpoint}", json=payload, timeout=60) # Added timeout
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Error: Could not connect to the Flask backend at {FLASK_API_URL}. "
                 "Please ensure the backend (app.py) is running.")
        return {"error": "Backend not reachable."}
    except requests.exceptions.HTTPError as e:
        st.error(f"Error from Flask API ({e.response.status_code}): "
                 f"{e.response.json().get('error', 'Unknown API error')}")
        return {"error": "API request failed."}
    except requests.exceptions.RequestException as e:
        st.error(f"An unexpected request error occurred: {e}")
        return {"error": "Request error."}
    except json.JSONDecodeError:
        st.error("Error: Received an unreadable response from the backend.")
        return {"error": "Invalid JSON response."}

# --- Patient Chat Tab ---
with tab1:
    st.header("💬 Patient Chat")
    st.write("Ask any health-related question and get an AI-generated informative response.")
    
    # Input area for the user's question
    user_question = st.text_area("Your Question:", height=150, key="chat_input_area")
    
    # Button to trigger the AI response
    if st.button("Get AI Response", key="chat_button"):
        if user_question:
            with st.spinner("HealthAI is thinking..."):
                result = call_flask_api("chat", {"message": user_question})
                if "response" in result:
                    st.success("Response from HealthAI:")
                    st.markdown(result["response"])
                elif "error" in result:
                    st.error(f"Failed to get response: {result['error']}")
        else:
            st.warning("Please enter your question in the text area.")

# --- Disease Prediction Tab ---
with tab2:
    st.header("🔍 Disease Prediction")
    st.write("Enter your symptoms, and HealthAI will suggest potential conditions and next steps.")
    
    # Input for symptoms
    symptoms_input = st.text_area(
        "Describe your symptoms in detail (e.g., persistent headache, fatigue, mild fever, nausea):",
        height=200,
        key="symptoms_input_area"
    )
    
    st.subheader("Optional: Your Patient Profile")
    col_age, col_gender = st.columns(2)
    with col_age:
        age = st.number_input("Age:", min_value=1, max_value=120, value=30, key="age_input")
    with col_gender:
        gender = st.selectbox("Gender:", ["Male", "Female", "Other", "Prefer not to say"], key="gender_select")
    
    # Button to trigger disease prediction
    if st.button("Predict Potential Conditions", key="predict_button"):
        if symptoms_input:
            with st.spinner("Analyzing symptoms and predicting potential conditions..."):
                patient_profile = {"age": age, "gender": gender}
                result = call_flask_api("predict_disease", {"symptoms": symptoms_input, "profile": patient_profile})
                if "prediction" in result:
                    st.success("Potential Conditions & Recommendations from HealthAI:")
                    st.markdown(result["prediction"])
                elif "error" in result:
                    st.error(f"Failed to get prediction: {result['error']}")
        else:
            st.warning("Please describe your symptoms to get a prediction.")

# --- Treatment Plans Tab ---
with tab3:
    st.header("💊 Treatment Plans")
    st.write("Generate a personalized, evidence-based treatment plan for a diagnosed condition.")
    
    # Input for diagnosed condition
    diagnosed_condition = st.text_input("Enter your diagnosed medical condition:", key="condition_input")

    st.subheader("Optional: Your Patient Profile for Personalized Plan")
    col_age_tp, col_gender_tp = st.columns(2)
    with col_age_tp:
        age_tp = st.number_input("Age (for plan):", min_value=1, max_value=120, value=30, key="age_tp_input")
    with col_gender_tp:
        gender_tp = st.selectbox("Gender (for plan):", ["Male", "Female", "Other", "Prefer not to say"], key="gender_tp_select")
    
    # Button to generate treatment plan
    if st.button("Generate Treatment Plan", key="treatment_button"):
        if diagnosed_condition:
            with st.spinner("Generating personalized treatment plan..."):
                patient_profile_tp = {"age": age_tp, "gender": gender_tp}
                result = call_flask_api("generate_treatment_plan", {"condition": diagnosed_condition, "profile": patient_profile_tp})
                if "plan" in result:
                    st.success(f"Treatment Plan for {diagnosed_condition} from HealthAI:")
                    st.markdown(result["plan"])
                elif "error" in result:
                    st.error(f"Failed to generate plan: {result['error']}")
        else:
            st.warning("Please enter a diagnosed condition to generate a treatment plan.")

# --- Health Analytics Tab ---
with tab4:
    st.header("📊 Health Analytics")
    st.write("Visualize your health metrics and receive AI-generated insights and recommendations.")
    
    st.markdown("""
        <div style="background-color: #e0f2f7; color: #01579b; border: 1px solid #b3e5fc; border-radius: 5px; padding: 10px; margin-bottom: 20px;">
            <strong>How to Use:</strong> Provide a summary or list of your health metrics (e.g., "My average heart rate last week was 75 bpm, blood pressure 120/80, blood glucose 95 mg/dL"). The AI will analyze trends and offer insights.
        </div>
    """, unsafe_allow_html=True)

    health_metrics_input = st.text_area(
        "Enter your health metrics data (e.g., 'Heart rate: [70, 72, 68], Blood Pressure: [120/80, 125/82], Average Blood Glucose: 95 mg/dL'):",
        height=200,
        key="health_metrics_input_area"
    )

    st.subheader("Optional: Your Patient Profile for Context")
    col_age_ha, col_gender_ha = st.columns(2)
    with col_age_ha:
        age_ha = st.number_input("Age (for analytics):", min_value=1, max_value=120, value=30, key="age_ha_input")
    with col_gender_ha:
        gender_ha = st.selectbox("Gender (for analytics):", ["Male", "Female", "Other", "Prefer not to say"], key="gender_ha_select")

    if st.button("Get Health Insights", key="analytics_button"):
        if health_metrics_input:
            with st.spinner("Analyzing health data and generating insights..."):
                # For simplicity, we'll send the raw text for now.
                # In a more advanced app, you might parse this into structured data (e.g., JSON).
                patient_profile_ha = {"age": age_ha, "gender": gender_ha}
                payload_data = {"data": health_metrics_input, "profile": patient_profile_ha}
                
                result = call_flask_api("health_analytics_insights", payload_data)
                
                if "insights" in result:
                    st.success("Health Insights from HealthAI:")
                    st.markdown(result["insights"])
                elif "error" in result:
                    st.error(f"Failed to get insights: {result['error']}")
        else:
            st.warning("Please provide some health metrics to get insights.")

# --- Footer ---
st.markdown("---")
st.markdown("<sub>Built with Streamlit and powered by IBM Watson & Granite-13b-instruct-v2.</sub>", unsafe_allow_html=True)

