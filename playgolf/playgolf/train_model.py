import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("playgolf_data (1).csv")
df.columns = df.columns.str.strip().str.title()  # Normalize column names

# Initialize LabelEncoders
le_outlook = LabelEncoder()
le_temperature = LabelEncoder()
le_humidity = LabelEncoder()
le_windy = LabelEncoder()
le_target = LabelEncoder()

# Encode input features
df['Outlook'] = le_outlook.fit_transform(df['Outlook'])
df['Temperature'] = le_temperature.fit_transform(df['Temperature'])
df['Humidity'] = le_humidity.fit_transform(df['Humidity'])
df['Wind'] = le_windy.fit_transform(df['Wind'])

# Encode target
df['Playgolf'] = le_target.fit_transform(df['Playgolf'])

# Print classes to verify
print("Target classes:", le_target.classes_)

# Feature and target
X = df[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = df['Playgolf']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model, scaler, encoders
joblib.dump(model, 'playgolf.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_outlook, 'le_outlook.pkl')
joblib.dump(le_temperature, 'le_temperature.pkl')
joblib.dump(le_humidity, 'le_humidity.pkl')
joblib.dump(le_windy, 'le_windy.pkl')
joblib.dump(le_target, 'le_target.pkl')

print("✅ Model, scaler, and encoders saved successfully.")
