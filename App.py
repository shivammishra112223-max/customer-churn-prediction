import joblib
from flask import Flask, render_template, request
import pandas as pd

from tensorflow.keras.models import load_model



App = Flask(__name__)

model=load_model("customer_churn_ann.keras")
scaler=joblib.load("standard_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

@App.route("/")
def home():
    return render_template("Index.html")
@App.route("/predict", methods=["POST"])
def predict():

    gender = int(request.form["gender"])
    SeniorCitizen = int(request.form["SeniorCitizen"])
    Partner = int(request.form["Partner"])
    Dependents = int(request.form["Dependents"])
    tenure = float(request.form["tenure"])
    PhoneService = int(request.form["PhoneService"])
    PaperlessBilling = int(request.form["PaperlessBilling"])
    MonthlyCharges = float(request.form["MonthlyCharges"])
    TotalCharges = float(request.form["TotalCharges"])

    InternetService = request.form["InternetService"]
    OnlineSecurity = request.form["OnlineSecurity"]
    OnlineBackup = request.form["OnlineBackup"]
    DeviceProtection = request.form["DeviceProtection"]
    TechSupport = request.form["TechSupport"]
    StreamingTV = request.form["StreamingTV"]
    StreamingMovies = request.form["StreamingMovies"]
    MultipleLines = request.form["MultipleLines"]
    Contract = request.form["Contract"]
    PaymentMethod = request.form["PaymentMethod"]

    # यहाँ से पूरा code 4 spaces अंदर होगा
    new_customer = pd.DataFrame({
        ...
    })



    new_customer = pd.DataFrame({

    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "PaperlessBilling": [PaperlessBilling],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges],

    "StreamingTV": [StreamingTV],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaymentMethod": [PaymentMethod]

})

# ---------------- Feature Engineering ----------------

    new_customer["LongTermCustomer"] = (
    new_customer["tenure"] > 24
).astype(int)

    new_customer["NewCustomer"] = (
    new_customer["tenure"] < 12
).astype(int)

# ---------------- One Hot Encoding ----------------

    new_customer = pd.get_dummies(
    new_customer,
    columns=[
        "StreamingTV",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ]
)
    
    
    
    new_customer = new_customer.reindex(
    columns=feature_columns,
    fill_value=0
)
    
    
    

# ---------------- Scaling ----------------

    scale_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

    new_customer[scale_columns] = scaler.transform(
    new_customer[scale_columns]
)

# ---------------- Prediction ----------------

    prediction = model.predict(new_customer)

    churn_probability = float(prediction[0][0]) * 100

# ---------------- Confidence ----------------

    if churn_probability >= 75:
        confidence = "High"
    elif churn_probability >= 50:
        confidence = "Medium"
    else:
        confidence = "Low"

# ---------------- Result ----------------




    if prediction[0][0] >= 0.5:
        result = "Customer Will Leave"
        probability = round(churn_probability, 2)
    else:
        result = "Customer Will Stay"
        probability = round(100 - churn_probability, 2)

# ---------------- HTML ----------------

    return render_template(
    "index.html",
    prediction=result,
    probability=probability,
    confidence=confidence
)

if __name__ == "__main__":
    App.run(debug=True)
    
    


   