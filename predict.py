import pandas as pd
import numpy as np
import os
import sys
import joblib

# Load the best model from the file
best_model = joblib.load("best_model.pkl")

# Get the path to the patient tables folder from the command line argument
patient_tables_path = sys.argv[1]

# Define a function to load and preprocess data from a folder
def load_and_preprocess_data(data_dir):
  data = []
  ids = []
  for file in sorted(os.listdir(data_dir)): # Sort the files by name
    # Get the patient id from the file name
    patient_id = file.split(".")[0]
    # Load the patient data from the file
    patient_data = pd.read_csv(os.path.join(data_dir, file), sep='|')
    # Fill missing values with forward fill method
    patient_data = patient_data.fillna(method='ffill')
    # Drop the sepsis label column from the data if it exists
    if "SepsisLabel" in patient_data.columns:
      patient_data = patient_data.drop(columns=["SepsisLabel"])
    # Use the last row of the data as the input for prediction
    data.append(patient_data.iloc[-1])
    # Append the patient id to the list
    ids.append(patient_id)
  return pd.DataFrame(data), np.array(ids)

# Load and preprocess data from the patient tables folder using the function
X, ids = load_and_preprocess_data(patient_tables_path)

# Predict on the data using the best model
y_pred = best_model.predict(X)

# Create a dataframe with ids and predictions
predictions = pd.DataFrame({"id": ids, "prediction": y_pred})

# Save the dataframe to a csv file in the current folder
predictions.to_csv("prediction.csv", index=False)