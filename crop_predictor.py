import numpy as np
import joblib

model = joblib.load("crop_recommendation_model.joblib")

feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def get_user_input():
    user_data = []
    print("Enter the values for the following features:")
    for feature in feature_names:
        value = float(input(f"{feature}: "))
        user_data.append(value)
    
    return np.array(user_data).reshape(1, -1)

user_input = get_user_input()

probabilities = model.predict_proba(user_input)

crop_labels = model.classes_

threshold = 0.75  

# Get crops with confidence scores above threshold
filtered_crops = [(crop_labels[idx], probabilities[0][idx]) for idx in np.argsort(probabilities[0])[::-1] if probabilities[0][idx] >= threshold]

# Display the crops above the threshold
if filtered_crops:
    print("\nRecommended Crops (Above 75% Confidence):")
    for crop, confidence in filtered_crops:
        print(f"{crop}: {confidence * 100:.2f}% confidence")
else:
    print("\nNo crops found with confidence above 75%")
