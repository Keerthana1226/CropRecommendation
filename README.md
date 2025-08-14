# Crop Recommendation System

This project predicts the most suitable crop to grow based on environmental conditions such as nitrogen, phosphorus, potassium levels, temperature, humidity, pH, and rainfall.  
It uses a machine learning model trained on agricultural data.

## Features
- Predict the best crop for given soil and weather parameters using RandomForest.
- Pre-trained model included (`crop_recommendation_model.joblib`).
- Jupyter notebooks for training and experimentation.

## Project Structure
Crop_Recommendation/

      │-- croppppp.ipynb                   # Notebook for testing and predictions
      │-- crop_predictor.py                # Script for running predictions
      │-- Crop_recommendation.csv          # Dataset
      │-- crop_recommendation_model.joblib # Trained ML model
      │-- training.ipynb                   # Model training notebook

## Requirements
Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```

1. Using Python Script
```bash
python crop_predictor.py
```
Follow the prompts to enter soil and weather values.

2. Using Jupyter Notebook

Open croppppp.ipynb or training.ipynb in Jupyter and run the cells.

