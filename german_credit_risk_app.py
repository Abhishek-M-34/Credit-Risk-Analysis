# SECTION 1: IMPORTS
import streamlit as st
import pickle
import numpy as np

# SECTION 2: MODEL LOADING
def load_artifacts():
    try:
        # Load model
        with open("logisticRegr.pkl", 'rb') as file:
            model = pickle.load(file)
        
        # Try to load scaler and PCA, but handle if they're missing or wrong
        scaler = None
        pca = None
        
        try:
            with open("scaler.pkl", 'rb') as file:
                scaler = pickle.load(file)
        except:
            st.warning("Scaler not found or invalid ")
            
        try:
            with open("pca.pkl", 'rb') as file:
                pca = pickle.load(file)
        except:
            st.warning("PCA not found or invalid ")
            
        return model, scaler, pca
        
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None

model, scaler, pca = load_artifacts()
# SECTION 3: ENCODING FUNCTIONS
def sexEncode(sex_input):
    sex_input = sex_input.lower().strip()
    if sex_input == "male":
        return 1
    else:
        return 0

def housingEncode(housing_input):
    housing_input = housing_input.lower().strip()
    if housing_input == "own":
        return 1
    elif housing_input == "free":
        return 0
    else:
        return 2

def savingAccountsEncode(savings_input):
    savings_input = savings_input.lower().strip()
    if savings_input == "little":
        return 1
    elif savings_input == "moderate":
        return 2
    elif savings_input == "quite rich":
        return 3
    elif savings_input == "rich":
        return 4
    elif savings_input == "":
        return 0
    else:
        return 0

def checkingAccountEncode(checking_input):
    checking_input = checking_input.lower().strip()
    if checking_input == "little":
        return 1
    elif checking_input == "moderate":
        return 2
    elif checking_input == "rich":
        return 3
    else:
        return 0

def purposeEncode(purpose_input):
    purpose_categories = ['radio/TV', 'education', 'furniture/equipment', 'car', 
                         'business', 'domestic appliances', 'repairs', 'vacation/others']
    purpose_values = [0.0] * len(purpose_categories)
    
    purpose_input = purpose_input.strip().lower()
    for i, category in enumerate(purpose_categories):
        if purpose_input == category.lower():
            purpose_values[i] = 1.0
            break
    
    return purpose_values

def resultOutput(result):
    return "bad" if result == 0 else "good"

# SECTION 4: PREDICTION FUNCTION
def riskPrediction(id_input, age_input, sex_input, job_input, housing_input,savings_input, checking_input, credit_input, duration_input, purpose_input):
    try:
        st.write("Starting prediction...")

        # Convert and encode inputs
        id_value = float(id_input)
        age_value = float(age_input)
        sex_value = sexEncode(sex_input)
        job_value = float(job_input)
        housing_value = housingEncode(housing_input)
        savings_value = savingAccountsEncode(savings_input)
        checking_value = checkingAccountEncode(checking_input)
        credit_value = float(credit_input)
        duration_value = float(duration_input)
        purpose_values = purposeEncode(purpose_input)

        # Prepare input vector
        input_data = np.array([[id_value, age_value, sex_value, job_value, housing_value,savings_value, checking_value, credit_value, duration_value] + purpose_values])

        st.write("Raw input features:", input_data)

        # Check if scaler and PCA are valid objects with transform method
        if scaler is None or not hasattr(scaler, 'transform'):
            return "Error: Scaler not available or invalid. Please check scaler.pkl file.", None
            
        if pca is None or not hasattr(pca, 'transform'):
            return "Error: PCA not available or invalid. Please check pca.pkl file.", None

        # Proper preprocessing pipeline
        scaled_data = scaler.transform(input_data)
        st.write("Scaled features:", scaled_data)
        
        pca_data = pca.transform(scaled_data)
        st.write("PCA transformed features:", pca_data)
        
        # Make prediction
        prediction = model.predict(pca_data)
        probabilities = model.predict_proba(pca_data)
        predicted_risk = int(prediction[0])
        confidence_good = probabilities[0][1]

        st.write("Raw Prediction Output:", prediction)
        st.write("Prediction Probabilities:", probabilities)
        st.write(f"Confidence in 'good': {confidence_good:.4f}")

        return predicted_risk, confidence_good

    except Exception as e:
        return f"Prediction Error: {e}", None

# SECTION 5: MAIN APPLICATION
def main():
    st.title("Credit Risk Analysis")

    # Input fields (same as before)
    id_input = st.number_input('Enter Id', step=1, value=1)
    age_input = st.number_input('Enter Age', min_value=18, max_value=100, step=1, value=35)
    sex_input = st.text_input('Enter Gender (Male / Female)', value='Male')
    job_input = st.number_input('Enter Job (0,1,2,3)', min_value=0, max_value=3, step=1, value=2)
    housing_input = st.text_input('Enter Housing Type (own, free, rent)', value='own')
    savings_input = st.text_input("Enter Savings Accounts ('little', 'quite rich', 'rich', 'moderate')", value='moderate')
    checking_input = st.text_input("Enter Checking Accounts ('little', 'moderate', 'rich')", value='moderate')
    credit_input = st.number_input('Enter Credit Amount', min_value=0.0, step=100.0, value=2000.0)
    duration_input = st.number_input('Enter Duration', min_value=0, step=1, value=12)
    purpose_input = st.selectbox("Enter Purpose",('radio/TV', 'education', 'furniture/equipment', 'car','business', 'domestic appliances', 'repairs', 'vacation/others'))

    if st.button('Predict Risk'):
        # Validation checks
        if id_input == 0:
            st.warning("Please enter your ID first.")
        elif age_input == 0:
            st.warning("Please enter your age first.")
        elif sex_input.strip() == "":
            st.warning("Please enter your gender first.")
        elif housing_input.strip() == "":
            st.warning("Please enter your housing detail first.")
        elif credit_input == 0:
            st.warning("Please enter your credit amount first.")
        elif duration_input == 0:
            st.warning("Please enter duration first.")
        else:
            result, confidence = riskPrediction(id_input, age_input, sex_input, job_input, housing_input,savings_input, checking_input, credit_input, duration_input, purpose_input)
    
            if isinstance(result, str) and (result.startswith("Error") or result.startswith("Prediction Error")):
                st.error(result)
            else:
                result_output = resultOutput(result)
                
                # Display results with confidence
                if result_output == "good":
                    st.success(f"Credit Risk Prediction: {result_output.upper()}")
                    st.metric("Confidence", f"{confidence:.1%}")
                else:
                    st.error(f"Credit Risk Prediction: {result_output.upper()}")
                    st.metric("Confidence", f"{confidence:.1%}")


if __name__ == '__main__':
    main()
