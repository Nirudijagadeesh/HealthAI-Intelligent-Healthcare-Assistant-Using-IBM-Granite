

```markdown
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

```

project/
│
├── app.py                # Flask backend
├── .env                  # IBM credentials (secure)
├── requirements.txt      # Python dependencies
└── streamlit\_app.py      # Streamlit frontend (optional)

````

---

## ⚙️ Setup Instructions

### 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt
````

Typical requirements:

```text
Flask
requests
python-dotenv
```

### 2. 🔐 Environment Setup

Create a `.env` file with your IBM credentials:

```env
IBM_API_KEY=your_ibm_api_key
IBM_PROJECT_ID=your_project_id
```

> Do **not** commit `.env` to version control.

### 3. ▶️ Start the Flask Server

```bash
python app.py
```

Flask will run at `http://localhost:5000` by default.

---

## 🌐 API Endpoints

| Endpoint                     | Method | Purpose                                    |
| ---------------------------- | ------ | ------------------------------------------ |
| `/`                          | GET    | Confirms backend is running                |
| `/chat`                      | POST   | AI answers general health questions        |
| `/predict_disease`           | POST   | Suggests possible conditions from symptoms |
| `/generate_treatment_plan`   | POST   | Returns treatment plan based on condition  |
| `/health_analytics_insights` | POST   | Analyzes user metrics and gives insights   |

---

## 🔄 IBM Granite API Integration

**Model ID**: `ibm/granite-13b-instruct-v2`
**Endpoint URL**:

```
https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-29
```

### ✉️ Request Structure

```json
{
  "model_id": "ibm/granite-13b-instruct-v2",
  "input": "prompt text",
  "parameters": {
    "decoding_method": "greedy",
    "max_new_tokens": 500,
    "min_new_tokens": 50,
    "repetition_penalty": 1.1
  },
  "project_id": "your_project_id"
}
```

### 🔐 Headers

```python
headers = {
  "Authorization": f"Bearer {API_KEY}",
  "Content-Type": "application/json",
  "ML-Instance-ID": PROJECT_ID
}
```

---

## 🧪 Example Usage (with `curl`)

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What causes fatigue and dizziness?"}'
```

---

## ✅ Error Handling

* **Timeouts**: Gracefully handled with friendly messages.
* **Missing Fields**: Validated before sending to model.
* **API Failures**: Logged with status codes for debugging.

---

## ⚠️ Disclaimer

> This backend facilitates interaction with an AI language model for informational health purposes. It does **not** offer real medical diagnosis or treatment. Always consult a healthcare professional for medical concerns.

---

## 🧩 Dependencies

Install via `pip install -r requirements.txt`

```text
Flask
requests
python-dotenv
```

---

## 🤝 Acknowledgments

* IBM WatsonX / Granite AI team
* Flask and Python open-source community
* Streamlit for rapid frontend development

---

## 📜 License

This backend is intended for educational and prototype purposes only. Redistribution should properly credit IBM and the authors.

---

```

---

Would you like me to also generate a `requirements.txt` file for both frontend and backend?
```
