import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from pydantic import BaseModel


app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

class BankNote(BaseModel):
    variance: float 
    skewness: float 
    curtosis: float 
    entropy: float
@app.get('/')
def index():
    return {'message': 'Bank Note Classifier'}

@app.post('/predict')
def predict_banknote(data:BankNote):
    data = dict(data)
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="True Note"
    return {
        'prediction': prediction
    }



if __name__ == '__main__':
    uvicorn.run(app)
    
#uvicorn app:app --reload