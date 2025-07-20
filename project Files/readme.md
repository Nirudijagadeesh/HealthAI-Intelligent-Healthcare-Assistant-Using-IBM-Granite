# 🧠 HealthAI Backend - Flask API Server

This is the **Flask-based backend** for **HealthAI**, an intelligent healthcare assistant leveraging IBM’s Granite-13b-instruct-v2 model. The backend serves as an API layer to process medical-related inputs and interact with IBM Watson's LLM via secure HTTP requests.

---

## 🔁 Responsibilities

The Flask backend:

- Receives POST requests from the frontend (Streamlit).
- Constructs prompts based on user data.
- Sends prompts to IBM Granite-13b-instruct-v2 via HTTP.
- Returns AI-generated medical insights or recommendations.

---

## 🔧 Technologies Used

| Component        | Library / Tech        | Role                                   |
|------------------|------------------------|----------------------------------------|
| Web Server       | Flask                  | Handles API requests                   |
| AI Interaction   | HTTP `requests`        | Communicates with IBM Granite LLM      |
| Auth/Security    | `python-dotenv`        | Loads API keys securely                |
| JSON Handling    | `json`                 | Payload construction & parsing         |

---

## 📁 File Structure
project Executable files
