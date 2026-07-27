import joblib
from flask import Flask, render_template, request
import pandas as pd

from tensorflow.keras.models import load_model



App = Flask(__name__)

model=load_model("customer_churn_ann.keras")
scaler=joblib.load("standard_scaler.pkl")

@App.route("/")
def home():
    return render_template("Index.html")

@App.route("/predict", methods=["POST"])
def predict():
    
    CreditScore = float(request.form["CreditScore"])
    Age = float(request.form["Age"])
    Tenure = float(request.form["Tenure"])
    Balance = float(request.form["Balance"])
    NumOfProducts = float(request.form["NumOfProducts"])
    HasCrCard = int(request.form["HasCrCard"])
    IsActiveMember = int(request.form["IsActiveMember"])
    EstimatedSalary = float(request.form["EstimatedSalary"])
    Gender = int(request.form["Gender"])
    Geography = int(request.form["Geography"])
    
    
    new_customer = pd.DataFrame({

    "CreditScore":[CreditScore],
    "Age":[Age],
    "Tenure":[Tenure],
    "Balance":[Balance],
    "NumOfProducts":[NumOfProducts],
    "HasCrCard":[HasCrCard],
    "IsActiveMember":[IsActiveMember],
    "EstimatedSalary":[EstimatedSalary],
    "Gender":[Gender],
    "Geography":[Geography]

})
    
    scale_columns = [

    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary"

]

    new_customer[scale_columns] = scaler.transform(
    new_customer[scale_columns]
)
    
    prediction = model.predict(new_customer)

    churn_probability = round(prediction[0][0] * 100,2)

# Confidence
    if churn_probability >= 80:
        confidence = "High"
    elif churn_probability >= 60:
        confidence = "Medium"
    else:
        confidence = "Low"

# Result
    if prediction[0][0] >= 0.5:
        result = "Customer Will Leave"
        
    else:
        
        result = "Customer Will Stay"

# HTML
    return render_template(
    "index.html",
    prediction=result,
    probability=churn_probability,
    confidence=confidence
)
       
        
if __name__ == "__main__":
     App.run(debug=True)