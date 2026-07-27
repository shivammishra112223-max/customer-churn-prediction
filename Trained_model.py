import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import joblib




Data=pd.read_csv(r"C:\Users\shiva\Downloads\archive (11)\WA_Fn-UseC_-Telco-Customer-Churn.csv")
#print(Data.head(5))

#print(Data.isnull().sum())
Data.drop('customerID',axis=1,inplace=True)
#print(Data.columns)

#Data type chek 
#print(Data.dtypes)

Data['TotalCharges']=pd.to_numeric(Data['TotalCharges'],errors='coerce')
# print(Data.dtypes)

print(Data.isnull().sum())
Data['TotalCharges']=Data['TotalCharges'].fillna(Data['TotalCharges'].median())
print(Data.isnull().sum())
#Duplicate value chek

# print(Data.duplicated().sum())

#Outlier dedect
columns=['MonthlyCharges','TotalCharges']
for col in columns:
    Q1 = Data[col].quantile(0.25)
    Q3 = Data[col].quantile(0.75)
    
    IQR =   Q3 - Q1
    
    lower=Q1 -1.5*IQR
    upper=Q3 +1.5*IQR
    
    outleir=Data[(Data[col] < lower) |  (Data[col] > upper)]
    
    print(col,len(outleir))
    print(Data.head(5))
    
    
#Encoding 

Data['gender']=Data['gender'].map({"Male":1,"Female":0})
Data['Partner']=Data['Partner'].map({"Yes":1,"No":0})
Data['Dependents']=Data['Dependents'].map({'Yes':1,"No":0})
Data['PaperlessBilling']=Data['PaperlessBilling'].map({'Yes':1,"No":0})
Data['PhoneService']=Data['PhoneService'].map({'Yes':1,"No":0})
Data['Churn']=Data['Churn'].map({'Yes':1,"No":0})


Data=pd.get_dummies(Data,columns=['StreamingTV','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                                  'TechSupport','StreamingMovies','Contract','PaymentMethod'])


#Feature Enginnering

Data['LongTermCustomer']=(Data['tenure'] > 24).astype(int)
Data['NewCustomer']=(Data['tenure'] < 12).astype(int)

# Scaling

Scaler=StandardScaler()
Data[['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']]=Scaler.fit_transform(
    Data[['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']]
)

#print(Data.select_dtypes(include='object').columns)


X= Data.drop('Churn',axis=1)
Y=Data['Churn']


X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

model=Sequential()
model.add(Dense(64,activation='relu',input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add( Dropout(0.3))
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())
model.add( Dropout(0.3))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
earlys=EarlyStopping(
    monitor='val_loss',
    patience=20,
    
)



history=model.fit(
    X_train,
    Y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[earlys],
    
)

prediction=model.predict(X_test)
print(prediction[:10])

prediction=(prediction > 0.5).astype(int)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

Accuracy=accuracy_score(Y_test,prediction)
print("Accuracy",Accuracy)

Accuracy=precision_score(Y_test,prediction)
print("precision",Accuracy)

Accuracy=recall_score(Y_test,prediction)
print("recal",Accuracy)

model.save("customer_churn_ann.keras")
joblib.dump(Scaler, "standard_scaler.pkl")
