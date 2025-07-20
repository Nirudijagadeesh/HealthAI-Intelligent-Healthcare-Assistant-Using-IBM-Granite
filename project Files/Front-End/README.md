
# 🩺 HealthAI Frontend - Streamlit Interface

This is the **Streamlit frontend** for **HealthAI**, an intelligent healthcare assistant powered by IBM Watson and the Granite-13b-instruct-v2 model. The frontend offers a clean, user-friendly interface for interacting with the Flask backend and AI-powered medical services.

---

## 🚀 Features

### ✅ Patient Chat
Ask any health-related question in natural language and get an AI-generated, informative response. Responses include empathetic explanations and always suggest consulting a medical professional.

### ✅ Disease Prediction
Input your symptoms and receive potential medical condition suggestions along with next steps. Includes optional user profile fields (age and gender) for more personalized AI responses.

### ✅ Treatment Plans
Provide a diagnosed condition and receive a detailed, AI-generated treatment plan, including:
- Medication suggestions
- Lifestyle recommendations
- Follow-up testing
Always flagged as a non-professional AI recommendation.

### ✅ Health Analytics
Paste your health metrics (heart rate, BP, glucose, etc.) and get trend analysis and actionable insights powered by the AI model.

---

## 🧰 Technologies Used

| Component      | Tool/Library     | Description                                     |
|----------------|------------------|-------------------------------------------------|
| Frontend       | Streamlit        | Interactive UI for user input/output            |
| Backend Comm.  | `requests`       | Sends data to Flask backend                     |
| Data Handling  | JSON             | API payloads and structured health data         |
| Styling        | Streamlit HTML   | Custom disclaimers and alert boxes              |

---

## 📦 Setup Instructions

### 1. 🔧 Prerequisites
- Python 3.8+
- Flask backend running (`app.py`)
- `.env` file for API keys in backend folder

### 2. 📁 Project Structure

```

project/
│
├── streamlit\_app.py       # Frontend main app
├── app.py                 # Flask backend
├── .env                   # API credentials
└── requirements.txt       # All dependencies

````

### 3. ▶️ Run the Frontend

Make sure your Flask server is running on `http://localhost:5000` or change the URL accordingly.

```bash
streamlit run streamlit_app.py
````

By default, Streamlit will open the app in your web browser at `http://localhost:8501`.

---

## ⚙️ Configurable Environment Variables

If needed, set this environment variable to match the backend location:

```bash
export FLASK_API_URL="http://localhost:5000"
```

In Windows (CMD):

```cmd
set FLASK_API_URL=http://localhost:5000
```

---

## 📋 User Flow Summary

| Tab                   | User Input                        | Backend Endpoint             | AI Action                          |
| --------------------- | --------------------------------- | ---------------------------- | ---------------------------------- |
| 💬 Patient Chat       | Natural-language health question  | `/chat`                      | LLM generates informative response |
| 🔍 Disease Prediction | Symptoms, age, gender             | `/predict_disease`           | Predicts possible conditions       |
| 💊 Treatment Plans    | Diagnosed condition, profile info | `/generate_treatment_plan`   | Creates treatment strategy         |
| 📊 Health Analytics   | Health metrics (text or list)     | `/health_analytics_insights` | Analyzes data & gives insights     |

---

## ⚠️ Disclaimer

> HealthAI is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified medical provider for health decisions. All AI responses are for **informational purposes only**.

---

## 📜 License

This project is for educational and demonstrative purposes. Attribution to IBM Watson and the open-source libraries used is required if redistributed.

---

## 🤝 Acknowledgments

* IBM WatsonX & Granite LLM
* Streamlit Community
* Flask and Python Open Source Contributors

---

```

---

Would you like a corresponding `README.md` file for the backend (`app.py`) as well?
```

