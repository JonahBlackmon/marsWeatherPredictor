import joblib
import numpy as np
import pandas as pd, urllib.request, json, requests

# Load the models
model_min = joblib.load('min_temp_model.pkl')
model_max = joblib.load('max_temp_model.pkl')

# Gets the data from NASA's API
with urllib.request.urlopen('https://mars.nasa.gov/rss/api/?feed=weather&category=msl&feedtype=json') as url:
    data = json.load(url)

# Makes our data frame based on the soles
df = pd.DataFrame(data["soles"])

# Only getting data related to terrestrial date for labeling, min/max temp, current season, and pressure
df = df[['terrestrial_date', 'min_temp', 'max_temp']]

# Removes blank entries
for x in range(len(df)) :
    if (df['max_temp'][x] == '--'):
        df.drop(x, inplace=True)

# Labels these columns as ints
df["min_temp"] = pd.to_numeric(df["min_temp"], errors="coerce")
df["max_temp"] = pd.to_numeric(df["max_temp"], errors="coerce")



# Define column names (make sure these match the training data feature names)
columns = ['min_temp', 'max_temp']

new_features = [[df['min_temp'][0], df['max_temp'][0]]]

# Convert new_features to a DataFrame
new_features_df = pd.DataFrame(new_features, columns=columns)

# Make predictions
predicted_min = model_min.predict(new_features_df)
predicted_max = model_max.predict(new_features_df)

print(f"""Predicted Values based on {df['terrestrial_date'][0]}:\n 
Predicted min temp: {predicted_min[0]}\n 
Predicted max temp: {predicted_max[0]}""")