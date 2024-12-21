from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained Stacking Classifier model
model = joblib.load("stacking_classifier_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define the input data model
class ModelInput(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Sex: str
    Embarked: str

# Map categorical features to numeric codes for prediction
def preprocess_input(data: ModelInput):
    sex_map = {"male": 0, "female": 1}
    embarked_map = {"C": 0, "Q": 1, "S": 2}

    # Calculate family_size
    family_size = data.SibSp + data.Parch + 1

    return np.array([
        data.Pclass,
        data.Age,
        data.SibSp,
        data.Parch,
        data.Fare,
        sex_map.get(data.Sex, -1),  # Handle unexpected values
        embarked_map.get(data.Embarked, -1),  # Handle unexpected values
        family_size
    ]).reshape(1, -1)

# Define the prediction route
@app.post("/predict")
async def predict(input_data: ModelInput):
    try:
        # Preprocess input
        processed_data = preprocess_input(input_data)
        
        # Check for invalid inputs
        if -1 in processed_data:
            return {"error": "Invalid input values for Sex or Embarked. Please use 'male', 'female' for Sex and 'C', 'Q', 'S' for Embarked."}
        
        # Make prediction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data).max()

        # Translate prediction to human-readable labels
        result = "Survived" if prediction[0] == 1 else "Not Survived"

        return {
            "prediction": result # Include family_size in response
            }
        
    except Exception as e:
        return {"error": f"Server Error: {str(e)}"}
