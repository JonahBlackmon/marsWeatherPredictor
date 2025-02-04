import pandas as pd, urllib.request, json, requests
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import joblib

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

# Sets the terrestrial_date to our datetime
df['terrestrial_date'] = pd.to_datetime(df['terrestrial_date'])
df.set_index('terrestrial_date', inplace=True)

# Labels these columns as ints
df["min_temp"] = pd.to_numeric(df["min_temp"], errors="coerce")
df["max_temp"] = pd.to_numeric(df["max_temp"], errors="coerce")


# Initializes tomorrow's min/max as separate comparable data
df['tomorrow_min'] = df['min_temp'].shift(1)
df['tomorrow_max'] = df['max_temp'].shift(1)



# Ensures there are no blanks
df = df.dropna()

# Sorts based on datetime
df = df.sort_index()

pd.set_option('display.max_rows', None)

features = df[['min_temp', 'max_temp']]
target_min = df['tomorrow_min']
target_max = df['tomorrow_max']

tscv = TimeSeriesSplit(n_splits = 5)

for train_index, test_index in tscv.split(features):
    feature_train, feature_test = features.iloc[train_index], features.iloc[test_index]
    train_min, test_min = target_min.iloc[train_index], target_min.iloc[test_index]
    train_max, test_max = target_max.iloc[train_index], target_max.iloc[test_index]

    # Train separate models for min_temp and max_temp
    model_min = LinearRegression().fit(feature_train, train_min)
    model_max = LinearRegression().fit(feature_train, train_max)
    # Predict
    pred_min = model_min.predict(feature_test)
    pred_max = model_max.predict(feature_test)

    # Evaluate using Mean Absolute Error (MAE)
    mae_min = mean_absolute_error(test_min, pred_min)
    mae_max = mean_absolute_error(test_max, pred_max)

    mae_min_scores.append(mae_min)
    mae_max_scores.append(mae_max)

# Save the trained model to pkl file
joblib.dump(model_min, 'min_temp_model.pkl')
joblib.dump(model_max, 'max_temp_model.pkl')
