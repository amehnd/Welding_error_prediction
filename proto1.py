import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("predictive_maintenance_welding.csv")

print("First few rows of the dataset:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDataset statistics:")
print(df.describe())


df.drop(columns=["Date"], inplace=True) #date not useful for modelling hence dropped

X = df.drop(columns=["Failure"])
y = df["Failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Scaling the data 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#training the model

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Knowing what feature affects the decision most by visualizing feature importance through bar graph 
feature_importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(8, 4))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Predicting Machine Failure")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()


joblib.dump(model, "welding_failure_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel and scaler saved successfully.")


                                                                #First batch testing 

# new_data = np.array([[210, 35, 75]])  # Voltage, Current, Temperature
# new_data_scaled = scaler.transform(new_data)
# prediction = model.predict(new_data_scaled)

# print("\nPrediction for new data:")
# print("Failure Likely – Maintenance Needed" if prediction[0] == 1 else "No Failure – No Maintenance Needed")


                                                                #Second batch testing

# Batch of new data points
test_data = pd.DataFrame([
    [210, 35, 75],
    [198, 30, 85],
    [250, 50, 90],
    [180, 20, 65],
    [205, 32, 110],
    [195, 29, 70],
    [240, 48, 68],
    [200, 31, 72],
    [220, 40, 95],
    [190, 33, 60],
], columns=["Voltage (V)", "Current (A)", "Temperature (°C)"])

# Scale the data
test_data_scaled = scaler.transform(test_data)

# Predict each row
predictions = model.predict(test_data_scaled)

# Output results
print("\nBatch Prediction Results:")
for i, result in enumerate(predictions):
    print(f"Data Point {i+1}: {'⚠️ Maintenance Required' if result == 1 else '✅ No Maintenance Needed'}")
