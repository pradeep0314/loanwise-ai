# LoanWise AI — Full Applicant Assessment Demo

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install packages: `pip install -r requirements.txt`
4. Start the app: `python app.py`
5. Open `http://127.0.0.1:5000`.

## Included
- Responsive multi-section loan application form.
- Personal, employment, active-loan, repayment-history, credit-profile and requested-loan fields.
- Flask backend and Random Forest model.
- SHAP feature explanations when the installed SHAP version is compatible.
- Render deployment configuration.

## Important model scope
The included synthetic CSV has exactly these eight input columns:
`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`,
`Credit_History`, `Education`, `Self_Employed`, and `Property_Area`.

Only these eight values are sent to `/predict` and used by the current model. The additional personal, active-loan, repayment and credit-profile fields are currently form/demo information only. To make them influence the model, obtain a suitable labelled dataset containing those features, then update the training pipeline and validate it properly. Do not claim that these extra fields affect the current prediction.

## Privacy
Do not enter real Aadhaar or PAN values. Those fields are demo-only. The browser JavaScript excludes identity and other non-model fields from the prediction request. This application does not save applicant records.

## Responsible use
This is a student demonstration, not a production lending tool. The included dataset is synthetic; its test accuracy is not evidence of real-world performance. Do not use it for actual credit decisions. A real deployment would require lawful data collection, security/privacy controls, representative independent validation, fairness and subgroup testing, compliance review and qualified human oversight. SHAP explains model behavior; it does not prove fairness or legal validity.

## Deploy on Render
Push this project folder to your GitHub repository. In Render, use:
- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Health check path: `/`
