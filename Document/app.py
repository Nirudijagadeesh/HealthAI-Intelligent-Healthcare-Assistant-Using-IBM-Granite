# app.py - Flask Backend for HealthAI

import os
import json
import requests # Import the requests library for direct HTTP calls
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# No longer importing from ibm_generative_ai as we are using direct HTTP requests.
# The previous imports (ibm_generative_ai.model_utils, ibm_generative_ai.inference, etc.)
# are removed as they are specific to the IBM SDK.

# Load environment variables from a .env file (for secure API key management)
# Create a file named .env in the same directory as app.py with your credentials:
# IBM_API_KEY="YOUR_ACTUAL_IBM_CLOUD_API_KEY"
# IBM_PROJECT_ID="YOUR_ACTUAL_WATSONX_AI_PROJECT_ID"
load_dotenv()

app = Flask(__name__)

# --- Retrieve IBM API Key and Project ID securely ---
# These variables should be set in your environment or in a .env file.
# DO NOT hardcode them directly in your application for security reasons.
ibm_api_key = os.getenv("IBM_API_KEY")
ibm_project_id = os.getenv("IBM_PROJECT_ID")

# Basic check to ensure API key and project ID are available
if not ibm_api_key or not ibm_project_id:
    print("WARNING: IBM_API_KEY and/or IBM_PROJECT_ID environment variables are not set.")
    print("Please set them up correctly before running the application.")
    # In a production app, you might want to raise an exception or handle this more robustly.

# --- Helper function to interact with IBM Granite-13b-instruct-v2 model using direct HTTP requests ---
def get_granite_response(prompt_text: str) -> str:
    """
    Sends a given prompt text to the IBM Granite-13b-instruct-v2 model
    via direct HTTP POST request and returns the generated text response.
    """
    if not ibm_api_key or not ibm_project_id:
        return "Error: IBM API Key or Project ID is not configured."

    # Define the API endpoint for Granite-13b-instruct-v2.
    # This URL is typically found in the IBM WatsonX.ai documentation or by inspecting SDK calls.
    # Ensure this URL is correct for your region (e.g., 'us-south').
    API_URL = "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-29"
    MODEL_ID = "ibm/granite-13b-instruct-v2" # This is the model ID used in the watsonx.ai API

    headers = {
        "Authorization": f"Bearer {ibm_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "ibm-metrics-api-token": ibm_api_key, # Sometimes required for metrics, depends on setup
        "ML-Instance-ID": ibm_project_id # This header links the request to your watsonx.ai project
    }

    # Construct the payload according to IBM Granite API specifications
    payload = {
        "model_id": MODEL_ID,
        "input": prompt_text,
        "parameters": {
            "decoding_method": "greedy", # "greedy" for deterministic output, "sample" for more creative
            "max_new_tokens": 500,       # Maximum tokens to generate
            "min_new_tokens": 50,        # Minimum tokens to generate
            "repetition_penalty": 1.1    # Penalize repeated tokens
        },
        "project_id": ibm_project_id # Project ID also often required in the body
    }

    try:
        # Make the POST request to the IBM Granite API
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        
        # Parse the JSON response
        response_data = response.json()
        
        # The exact path to the generated text might vary slightly based on API version.
        # This is a common structure for IBM's text generation APIs.
        if response_data and "results" in response_data and len(response_data["results"]) > 0:
            return response_data["results"][0].get("generated_text", "No text generated.")
        else:
            print(f"Unexpected API response structure: {response_data}")
            return "An unexpected response format was received from the AI."

    except requests.exceptions.Timeout:
        print("Error: Request to IBM Granite API timed out.")
        return "The AI took too long to respond. Please try again."
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to IBM Granite API. Check network and API URL.")
        return "Could not connect to the AI service. Please check your network connection."
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error calling IBM Granite API: {e.response.status_code} - {e.response.text}")
        return f"An API error occurred: {e.response.status_code}. Please check your credentials or API usage."
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from API response: {response.text}")
        return "Received an unreadable response from the AI service."
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred while interacting with IBM Granite API: {e}")
        return "An internal error occurred while processing your request with the AI."

# --- Flask Routes ---

@app.route('/')
def home():
    """Simple home route to confirm the Flask app is running."""
    return "HealthAI Flask Backend is running!"

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    """
    Handles chat requests, sending user messages to Granite and returning AI's response.
    Expected JSON input: {"message": "Your health question"}
    """
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Craft a specific prompt for the Patient Chat scenario
    # This prompt guides the AI to act as a helpful, informative, and cautious assistant.
    prompt_for_granite = (
        f"As a compassionate and informative healthcare assistant, please provide a clear "
        f"and concise answer to the following health-related question. Always ensure "
        f"to mention that this information is for general knowledge and not a substitute "
        f"for professional medical advice. Question: '{user_message}'"
    )
    
    ai_response = get_granite_response(prompt_for_granite)
    
    return jsonify({"response": ai_response})

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    """
    Handles disease prediction requests based on symptoms and patient profile.
    Expected JSON input: {"symptoms": "description of symptoms", "profile": {"age": N, "gender": "X"}}
    """
    data = request.get_json()
    symptoms = data.get('symptoms')
    patient_profile = data.get('profile', {}) # Default to empty dict if no profile is provided

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    # Craft a specific prompt for the Disease Prediction scenario
    # Instruct the AI to provide potential conditions, likelihoods, and next steps,
    # always emphasizing professional medical consultation.
    profile_str = json.dumps(patient_profile) # Convert profile dict to string for prompt
    prompt_for_granite = (
        f"A user describes the following symptoms: '{symptoms}'. "
        f"The user's profile is: {profile_str}. "
        f"Based on this information, list potential conditions, their approximate likelihood, "
        f"and suggest general next steps. Emphasize that a professional medical diagnosis "
        f"is essential and this is for informational purposes only."
    )
    
    ai_response = get_granite_response(prompt_for_granite)
    
    return jsonify({"prediction": ai_response})

@app.route('/generate_treatment_plan', methods=['POST'])
def generate_treatment_plan():
    """
    Handles requests for generating personalized treatment plans.
    Expected JSON input: {"condition": "Diagnosed condition", "profile": {...}}
    """
    data = request.get_json()
    condition = data.get('condition')
    patient_profile = data.get('profile', {})

    if not condition:
        return jsonify({"error": "No condition provided"}), 400

    profile_str = json.dumps(patient_profile)
    prompt_for_granite = (
        f"Based on the diagnosed condition '{condition}' and the patient profile: {profile_str}, "
        f"create a comprehensive, evidence-based treatment plan. Include suggestions for "
        f"medications, lifestyle modifications, and follow-up testing. State clearly that "
        f"this plan is an AI-generated suggestion and should be reviewed by a medical doctor."
    )

    ai_response = get_granite_response(prompt_for_granite)

    return jsonify({"plan": ai_response})

@app.route('/health_analytics_insights', methods=['POST'])
def health_analytics_insights():
    """
    Provides AI-generated insights based on provided health metrics data.
    Expected JSON input: {"data": {"metric_name": [value1, value2, ...], ...}, "profile": {...}}
    """
    data = request.get_json()
    health_data = data.get('data') # Example: {"heart_rate": [70, 72, 68], "blood_pressure": ["120/80", "125/82"]}
    patient_profile = data.get('profile', {})

    if not health_data:
        return jsonify({"error": "No health data provided"}), 400

    data_str = json.dumps(health_data)
    profile_str = json.dumps(patient_profile)
    prompt_for_granite = (
        f"Analyze the following health metrics: {data_str} for a patient with profile: {profile_str}. "
        f"Identify any potential trends, concerns, or areas for improvement. "
        f"Provide actionable recommendations. Remind the user to consult a healthcare professional for interpretation."
    )

    ai_response = get_granite_response(prompt_for_granite)

    return jsonify({"insights": ai_response})


# --- Run the Flask app ---
if __name__ == '__main__':
    # Flask runs on port 5000 by default.
    # debug=True allows for automatic reloads on code changes (for development only).
    # For production, use a WSGI server like Gunicorn (e.g., gunicorn -w 4 app:app).
    app.run(debug=True, port=5000)
