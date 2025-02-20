import streamlit as st
import numpy as np
import pickle

# importing model
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Create Streamlit interface
def main():
    st.title("Crop Recommendation System")
    
    # Creating input fields for features
    Nitrogen = st.number_input('Nitrogen (N)', min_value=0.0, step=0.1)
    Phosphorus = st.number_input('Phosphorus (P)', min_value=0.0, step=0.1)
    Potassium = st.number_input('Potassium (K)', min_value=0.0, step=0.1)
    Temperature = st.number_input('Temperature', min_value=0.0, step=0.1)
    Humidity = st.number_input('Humidity', min_value=0.0, step=0.1)
    Ph = st.number_input('pH', min_value=0.0, step=0.1)
    Rainfall = st.number_input('Rainfall', min_value=0.0, step=0.1)

    # Creating a prediction button
    if st.button('Predict'):
        # Collecting feature list from inputs
        feature_list = [Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Scaling the features
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        # Making the prediction
        prediction = model.predict(final_features)

        # Define the crop dictionary
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        # Displaying the result
        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            st.success(f"{crop} is the best crop to be cultivated with the provided data.")
        else:
            st.error("Sorry, we could not determine the best crop to be cultivated with the provided data.")

if __name__ == '__main__':
    main()
