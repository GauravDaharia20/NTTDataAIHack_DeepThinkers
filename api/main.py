from fastapi import FastAPI
import tensorflow_text as text
import tensorflow as tf
import uvicorn
import numpy as np
from pydantic import BaseModel
from joblib import load

from typing import Optional

url = 'http://localhost:8000/predict'

class get_symtom(BaseModel):
    symtom:str



class Predict:
    def predict_class(self,reviews):
        '''predict class of input text
        Args:
        - reviews (list of strings)
        Output:
        - class (list of int)
      '''
        return [np.argmax(pred) for pred in model.predict(reviews)]
    
    def actual_value(self,values):
        val_back={15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
    
        ans = []
        for i in values:
            ans.append(val_back[i])
        return ans
    
    def convert_value(self,values):
        return [np.argmax(i) for i in values]
            
    
    

app = FastAPI()

#encode = load("../DiseasesData/embed.joblib")
model = tf.keras.models.load_model("C:/Users/PC-DELL/Datasets/DiseasesData/model/model_1")
pred = Predict()

@app.get("/predict")
async def Prediction(sym:str):
    prec = {'Drug Reaction': 'stop irritation consult nearest hospital stop taking drug follow up',
 'Malaria': 'Consult nearest hospital avoid oily food avoid non veg food keep mosquitos out',
 'Allergy': 'apply calamine cover area with bandage  use ice to compress itching',
 'Hypothyroidism': 'reduce stress exercise eat healthy get proper sleep',
 'Psoriasis': 'wash hands with warm soapy water stop bleeding using pressure consult doctor salt baths',
 'GERD': 'avoid fatty spicy food avoid lying down after eating maintain healthy weight exercise',
 'Chronic cholestasis': 'cold baths anti itch medicine consult doctor eat healthy',
 'hepatitis A': 'Consult nearest hospital wash hands through avoid fatty spicy food medication',
 'Osteoarthristis': 'acetaminophen consult nearest hospital follow up salt baths',
 '(vertigo) Paroymsal  Positional Vertigo': 'lie down avoid sudden change in body avoid abrupt head movment relax',
 'Hypoglycemia': 'lie down on side check in pulse drink sugary drinks consult doctor',
 'Acne': 'bath twice avoid fatty spicy food drink plenty of water avoid too many products',
 'Diabetes ': 'have balanced diet exercise consult doctor follow up',
 'Impetigo': 'soak affected area in warm water use antibiotics remove scabs with wet compressed cloth consult doctor',
 'Hypertension ': 'meditation salt baths reduce stress get proper sleep',
 'Peptic ulcer diseae': 'avoid fatty spicy food consume probiotic food eliminate milk limit alcohol',
 'Dimorphic hemmorhoids(piles)': 'avoid fatty spicy food consume witch hazel warm bath with epsom salt consume alovera juice',
 'Common Cold': 'drink vitamin c rich drinks take vapour avoid cold food keep fever in check',
 'Chicken pox': 'use neem in bathing  consume neem leaves take vaccine avoid public places',
 'Cervical spondylosis': 'use heating pad or cold pack exercise take otc pain reliver consult doctor',
 'Hyperthyroidism': 'eat healthy massage use lemon balm take radioactive iodine treatment',
 'Urinary tract infection': 'drink plenty of water increase vitamin c intake drink cranberry juice take probiotics',
 'Varicose veins': 'lie down flat and raise the leg high use oinments use vein compression dont stand still for long',
 'AIDS': 'avoid open cuts wear ppe if possible consult doctor follow up',
 'Paralysis (brain hemorrhage)': 'massage eat healthy exercise consult doctor',
 'Typhoid': 'eat high calorie vegitables antiboitic therapy consult doctor medication',
 'Hepatitis B': 'consult nearest hospital vaccination eat healthy medication',
 'Fungal infection': 'bath twice use detol or neem in bathing water keep infected area dry use clean cloths',
 'Hepatitis C': 'Consult nearest hospital vaccination eat healthy medication',
 'Migraine': 'meditation reduce stress use poloroid glasses in sun consult doctor',
 'Bronchial Asthma': 'switch to loose cloothing take deep breaths get away from trigger seek help',
 'Alcoholic hepatitis': 'stop alcohol consumption consult doctor medication follow up',
 'Jaundice': 'drink plenty of water consume milk thistle eat fruits and high fiberous food medication',
 'Hepatitis E': 'stop alcohol consumption rest consult doctor medication',
 'Dengue': 'drink papaya leaf juice avoid fatty spicy food keep mosquitos away keep hydrated',
 'Hepatitis D': 'consult doctor medication eat healthy follow up',
 'Heart attack': 'call ambulance chew or swallow asprin keep calm',
 'Pneumonia': 'consult doctor medication rest follow up',
 'Arthritis': 'exercise use hot and cold therapy try acupuncture massage',
 'Gastroenteritis': 'stop eating solid food for while try taking small sips of water rest ease back into eating',
 'Tuberculosis': 'cover mouth consult doctor medication rest'}
    text = [sym]
    pval = [np.argmax(pred) for pred in model.predict(text)]
    acval = pred.actual_value(pval)
    return {"sentence":text,"prediction":acval[0],"precaution tip":prec[acval[0]]}
    
if __name__ == "__main__":
    uvicorn.run(app , host='localhost',port=8000)