from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
DATA_PATH = "data/loan_sample.csv"

def train_model():
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=["Loan_Status"]), df["Loan_Status"].astype(str)
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(handle_unknown="ignore"))])
    transformers=[]
    if numeric: transformers.append(("num",num_pipe,numeric))
    if categorical: transformers.append(("cat",cat_pipe,categorical))
    prep=ColumnTransformer(transformers)
    clf=RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced",min_samples_leaf=2)
    pipe=Pipeline([("prep",prep),("model",clf)])
    stratify=y if y.value_counts().min()>=2 else None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=stratify)
    pipe.fit(Xtr,ytr)
    return pipe, accuracy_score(yte,pipe.predict(Xte)), X.columns.tolist()

model, accuracy, FEATURES = train_model()

@app.route("/")
def home():
    return render_template("index.html", accuracy=f"{accuracy:.1%}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload=request.get_json(force=True)
        row={
          "ApplicantIncome":float(payload["ApplicantIncome"]),
          "CoapplicantIncome":float(payload["CoapplicantIncome"]),
          "LoanAmount":float(payload["LoanAmount"]),
          "Loan_Amount_Term":float(payload["Loan_Amount_Term"]),
          "Credit_History":int(payload["Credit_History"]),
          "Education":payload["Education"],
          "Self_Employed":payload["Self_Employed"],
          "Property_Area":payload["Property_Area"]
        }
        applicant=pd.DataFrame([row],columns=FEATURES)
        pred=str(model.predict(applicant)[0])
        probs=model.predict_proba(applicant)[0]
        classes=list(model.named_steps["model"].classes_)
        positive=next((c for c in classes if c.lower() in ["y","yes","approved","1","true"]),classes[-1])
        probability=float(probs[classes.index(positive)])
        # Transparent, human-readable input summary. SHAP is calculated where installed.
        explanations=[]
        try:
            import shap
            transformed=model.named_steps["prep"].transform(applicant)
            explainer=shap.TreeExplainer(model.named_steps["model"])
            sv=explainer.shap_values(transformed)
            ci=classes.index(positive)
            if isinstance(sv,list):
                vals=np.asarray(sv[ci]).reshape(-1)
            else:
                arr=np.asarray(sv)
                vals=arr[0,:,ci] if arr.ndim==3 else (arr[0] if arr.ndim==2 else arr.reshape(-1))
            names=model.named_steps["prep"].get_feature_names_out()
            pairs=sorted(zip(names,vals),key=lambda x:abs(float(x[1])),reverse=True)[:5]
            explanations=[{"feature":str(k).replace("num__","").replace("cat__","").replace("_"," "),
                           "impact":round(float(v),4)} for k,v in pairs]
        except Exception:
            explanations=[
              {"feature":"Credit history","impact":None},
              {"feature":"Applicant and co-applicant income","impact":None},
              {"feature":"Loan amount","impact":None}
            ]
        return jsonify({"prediction":pred,"probability":probability,"positive_label":positive,
                        "accuracy":accuracy,"explanations":explanations})
    except Exception as e:
        return jsonify({"error":str(e)}),400

if __name__=="__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
