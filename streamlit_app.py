import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("stacking_classifier_model.pkl")

# Mapping categorical features
sex_map = {"male": 0, "female": 1}
embarked_map = {"C": 0, "Q": 1, "S": 2}

# Title of the Streamlit App
st.title("Titanic Survival Prediction App")

# Sidebar for user inputs
st.sidebar.header("Passenger Details")
pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=0)
age = st.sidebar.slider("Age", min_value=0, max_value=100, value=28)
sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, value=0)
parch = st.sidebar.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, value=0)
fare = st.sidebar.number_input("Fare", min_value=0.0, step=0.1, value=30.0)
sex = st.sidebar.selectbox("Sex", ["male", "female"], index=0)
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"], index=0)

# Calculate FamilySize as SibSp + Parch + 1
family_size = sibsp + parch + 1

# Prediction button
if st.button("Predict Survival"):
    try:
        # Preprocess input
        input_data = np.array([
            pclass,
            sex_map[sex],
            age,
            sibsp,
            parch,
            fare,
            embarked_map[embarked],
            family_size
        ]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data).max()

        # Display results
        if prediction[0] == 1:
            st.success(f"The passenger is predicted to survive")
        else:
            st.error(f"The passenger is predicted NOT to survive")
    except Exception as e:
        st.error(f"An error occurred: {e}")
