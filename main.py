from fastapi import FastAPI
from pydantic import BaseModel,Field
import pandas as pd
from fastapi.staticfiles import StaticFiles

import joblib
from contextlib import asynccontextmanager



ml_model={}
@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model['model']=joblib.load('credit_risk_model.pkl')
    ml_model['threshold']=joblib.load('best_threshold.pkl')
    yield
    ml_model.clear() #gets clear from the memory once the owrk is done 
app=FastAPI(lifespan=lifespan)#model ko tabtak open rakhege jabtak server band nhi hota isliye ye function banaye hai 


class LoanApplication(BaseModel):
    #input column which user will fill
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


@app.post('/predict')
def predict(data :LoanApplication):#data is the object and loanapplication is the class
    input_df = pd.DataFrame([data.dict()])
    probability = ml_model['model'].predict_proba(input_df)[:, 1][0]

    prediction = int(probability >= ml_model["threshold"])

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": ml_model["threshold"],
        "Result": "High Risk" if prediction == 1 else "Low Risk"
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")



