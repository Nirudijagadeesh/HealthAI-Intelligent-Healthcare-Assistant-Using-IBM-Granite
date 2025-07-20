
### 📁 Full Project README: **HealthAI - Intelligent Healthcare Assistant**

---

## 🌐 Project Overview

**HealthAI** is an intelligent AI-powered healthcare assistant designed to offer:

* Patient Chat interaction for general medical queries.
* Disease prediction based on symptoms and patient profile.
* Personalized treatment planning.
* Health analytics based on user-provided health metrics.

⚠️ **Disclaimer**: HealthAI is *not* a replacement for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals.

---

## 📦 Folder Structure

```plaintext
HealthAI/
│
├── backend/
│   ├── app.py
│   ├── .env
│   ├── requirements.txt
│   └── README.md
│
├── frontend/
│   ├── streamlit.py
│   ├── requirements.txt
│   └── README.md
│
├── documentation/
│   ├── system_architecture.png
│   ├── user_manual.pdf
│   ├── API_reference.md
│   └── model_description.md
│
├── projectfiles/
│   ├── healthai_report.pdf
│   ├── presentation.pptx
│   └── prototype_results.xlsx
│
├── video_description/
│   ├── HealthAI_Demo.mp4
│   └── HealthAI_Explanation.txt
│
└── README.md   ← (You are here)
```

---

## ⚙️ Technologies Used

| Layer      | Technology                                         |
| ---------- | -------------------------------------------------- |
| Frontend   | Streamlit                                          |
| Backend    | Flask                                              |
| AI Model   | IBM Granite-13b-instruct-v2 (WatsonX)              |
| Language   | Python 3.10+                                       |
| Deployment | Localhost (or cloud via IBM Cloud, AWS, GCP, etc.) |

---

## 🧠 Features

### 1. 💬 Patient Chat

* Ask health-related questions.
* AI gives compassionate, informative, general advice.
* Emphasizes importance of consulting a doctor.

### 2. 🔍 Disease Prediction

* User enters symptoms.
* AI suggests possible conditions & recommendations.
* Uses demographic data (age, gender) to enhance accuracy.

### 3. 💊 Treatment Plans

* Generate an AI-based plan for a diagnosed condition.
* Includes medications, lifestyle changes, follow-ups.
* Clearly states the need for a doctor’s review.

### 4. 📊 Health Analytics

* User enters health metrics (BP, glucose, heart rate).
* AI analyzes trends and offers actionable insights.
* Reports generated in natural language.

---

## 🚀 Getting Started

### 🛠️ Prerequisites

* Python 3.10+
* Virtual environment (recommended)
* IBM WatsonX API credentials

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/HealthAI.git
cd HealthAI
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

---

## 🖥️ Backend Setup (Flask)

### 🔍 Navigate to backend/

```bash
cd backend
```

### 📄 Install Requirements

```bash
pip install -r requirements.txt
```

### 🔑 Add your `.env` file

Create a `.env` file with:

```ini
IBM_API_KEY=your_ibm_api_key_here
IBM_PROJECT_ID=your_ibm_project_id_here
```

### ▶️ Run Backend

```bash
python app.py
```

---

## 🌐 Frontend Setup (Streamlit)

### 🔍 Navigate to frontend/

```bash
cd ../frontend
```

### 📄 Install Requirements

```bash
pip install -r requirements.txt
```

### ▶️ Run Frontend

```bash
streamlit run streamlit.py
```

---

## 🔄 Backend ↔ Frontend Integration

Make sure your `.streamlit.py` in `frontend/` has:

```python
FLASK_API_URL = os.getenv("FLASK_API_URL", "http://localhost:5000")
```

Ensure the Flask server is running on port 5000 for the frontend to communicate.

---

## 📜 Backend README (`backend/README.md`)

**Contains**:

* API endpoint details
* IBM WatsonX configuration
* Prompt engineering strategy
* Error handling & logging practices

---

## 🧑‍💻 Frontend README (`frontend/README.md`)

**Contains**:

* Streamlit page breakdown
* Tab structure & UX elements
* Connection with Flask backend
* Exception handling strategies

---

## 📚 Documentation Folder (`/documentation`)

**Files**:

* `system_architecture.png`: Diagram of frontend, backend, model API.
* `API_reference.md`: Describes request/response formats.
* `user_manual.pdf`: Usage guide for end users.
* `model_description.md`: Details on IBM Granite-13b-instruct-v2 usage.

---

## 📁 Project Files Folder (`/projectfiles`)

**Files**:

* `healthai_report.pdf`: Full academic/technical report.
* `presentation.pptx`: Pitch deck / presentation material.
* `prototype_results.xlsx`: Evaluation data & metrics.

---

## 🎥 Video Description Folder (`/video_description`)

**Files**:

* `HealthAI_Demo.mp4`: Walkthrough of the system.
* `HealthAI_Explanation.txt`: Script for the video narration.

---

## 🔁 API Endpoints Overview

| Endpoint                     | Method | Description                        |
| ---------------------------- | ------ | ---------------------------------- |
| `/chat`                      | POST   | Responds to user health questions. |
| `/predict_disease`           | POST   | Predicts potential diseases.       |
| `/generate_treatment_plan`   | POST   | Suggests treatment plans.          |
| `/health_analytics_insights` | POST   | Provides health trend insights.    |

---

## 🧪 Example JSON Payloads

### `/chat`

```json
{
  "message": "What are the early symptoms of diabetes?"
}
```

### `/predict_disease`

```json
{
  "symptoms": "Fatigue, blurred vision, excessive thirst",
  "profile": {
    "age": 45,
    "gender": "Male"
  }
}
```

### `/generate_treatment_plan`

```json
{
  "condition": "Type 2 Diabetes",
  "profile": {
    "age": 55,
    "gender": "Female"
  }
}
```

### `/health_analytics_insights`

```json
{
  "data": {
    "heart_rate": [75, 80, 70],
    "blood_pressure": ["120/80", "125/82"],
    "glucose": [90, 95, 100]
  },
  "profile": {
    "age": 40,
    "gender": "Other"
  }
}
```

---

## 🧰 Error Handling

Each endpoint:

* Checks for required fields.
* Gracefully handles missing or malformed input.
* Returns user-friendly messages and logs internal errors for debugging.

---

## 🔐 Security Considerations

* IBM API Key and Project ID stored in `.env` file.
* No API keys are hardcoded.
* Can be secured via HTTPS when deployed.

---

## 📦 Future Improvements

* ✅ User authentication
* ✅ Feedback system for AI responses
* ✅ Admin dashboard
* ❌ Multi-language support
* ❌ Integration with wearable health devices
* ❌ Export reports as PDF

---

## ✅ Project Status

* ✅ Backend connected to IBM WatsonX
* ✅ Functional Streamlit frontend
* ✅ Core AI functionalities complete
* ✅ Documentation complete
* 🚧 Deployment scripts pending

---

## 👥 Team & Credits

| Role         | Name             |
| ------------ | ---------------- |
| Project Lead | \[Your Name]     |
| AI Engineer  | \[Teammate Name] |
| DevOps       | \[Teammate Name] |
| UX Designer  | \[Teammate Name] |

---

## 📩 Contact

* Email: `healthai-support@example.com`
* LinkedIn: \[Your LinkedIn]
* GitHub: \[Your GitHub]

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Would you like this full structure and README bundled as downloadable files? Or want any part customized (e.g., team credits, email, GitHub)?
