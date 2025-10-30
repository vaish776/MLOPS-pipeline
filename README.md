#  Continuous Model Validation Pipeline using MLOps

This project demonstrates how to build a **Continuous Model Validation Pipeline** — an essential concept in **MLOps** (Machine Learning Operations).  
It automates **training**, **validation**, **testing**, and **deployment** of a simple ML model using **scikit-learn**, **pytest**, **FastAPI**, and **GitHub Actions** for CI/CD.

---

##  Project Overview

Traditional ML models are usually trained once and manually evaluated.  
However, in real-world applications, data constantly changes — leading to **model drift** and reduced performance over time.  

To solve this, this project builds a **continuous validation workflow** that:
- Automatically **re-trains and validates** models.
- Runs **automated tests** to ensure performance.
- Deploys the model as an **API service** using FastAPI.
- Integrates with **GitHub Actions** for CI/CD.

---

##  Tech Stack

| Category | Tool/Library |
|-----------|---------------|
| Programming Language | Python 3 |
| ML Framework | scikit-learn |
| Model Saving | joblib |
| API Framework | FastAPI |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Environment | Python Virtual Environment (venv) |

---

##  Project Structure
mlops-pipeline/
│
├── src/
│ ├── train.py # Train and save the model
│ ├── validate.py # Validate model accuracy
│ └── api.py # FastAPI app for model predictions
│
├── tests/
│ └── test_model.py # Unit tests for model and validation
│
├── .github/
│ └── workflows/
│ └── ci.yml # GitHub Actions workflow for CI/CD
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation




---

## 🚀 How to Run the Project

### 1️⃣ Setup Environment
```bash
cd C:\Users\Vaishnavi\OneDrive\Desktop\mlops-pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt



2. Train the Model
python src/train.py


3. Validate the Model
python src/validate.py


4. Run Tests
pytest tests/ -v


5. Run FastAPI App
uvicorn src.api:app --reload



6.Then open:
 http://127.0.0.1:8000/docs

Example input for /predict:

{
  "features": [5.1, 3.5, 1.4, 0.2]
}


CI/CD (GitHub Actions)

Whenever you push code to GitHub, the CI/CD pipeline in .github/workflows/ci.yml will automatically:

Install dependencies

Run model training

Validate accuracy

Run pytest unit tests

This ensures that every commit keeps the model working correctly.




Use Case

This kind of pipeline is widely used in:

Finance → Fraud detection systems

Healthcare → Predictive diagnosis models

E-commerce → Recommendation engines

IoT → Predictive maintenance

It ensures models stay accurate, reliable, and continuously monitored.





Key Takeaways

Automates model lifecycle (train → validate → deploy)

Ensures reproducibility through CI/CD

Simplifies deployment using FastAPI

Demonstrates MLOps principles clearly for students and beginners


Output Example
✔️ API Running Message
{"status": "ok", "message": "Iris model is ready. Use POST /predict."}



Prediction Example

Input:

{"features": [5.1, 3.5, 1.4, 0.2]}


Output:

{"prediction": "setosa"}