# LoanWise AI — Loan Eligibility Website

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install dependencies: `pip install -r requirements.txt`
4. Start: `python app.py`
5. Visit http://127.0.0.1:5000

## Included
- Responsive landing page and applicant form
- Flask backend and Random Forest model trained on the included synthetic CSV
- Applicant-level prediction and probability
- SHAP feature contribution explanations (if SHAP is installed and compatible)
- Sample model test accuracy display

## Use your own data
Replace `data/loan_sample.csv` with a labelled CSV containing the same input columns and `Loan_Status` target, or update `train_model()` and the form fields in `app.py` / `templates/index.html` to match your schema.

## Responsible use
This is a student demonstration, not a production lending tool. The included dataset is synthetic and model accuracy is not evidence of real-world performance. Do not make real credit decisions using this prototype. Real deployment requires legal/compliance review, representative independent validation, fairness and subgroup testing, security/privacy controls, and qualified human review. SHAP explains model behavior; it does not establish fairness or legal validity.

## Deploy on Render
Push this folder to a GitHub repository, then in Render choose New + > Web Service, connect the repository, and use:
- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Health check path: `/`
Render provides the public URL after deployment.
