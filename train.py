#Train churn prediction model using synthetic churn dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

#Load Data
data = pd.read_csv('data/churn_data.csv')

#Feature Selection
features = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'num_support_calls']
X = data[features]
y = data['churn']

#Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Random Forest Classifier
# Initialize RandomForestClassifier with 100 trees (common default) and a fixed random state for reproducibility
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

# Calculate the accuracy of the model's predictions
accuracy = accuracy_score(y_test, y_pred)
# Calculate the area under the ROC curve for the model's predictions
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Model Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

#Save Model
import pickle
with open('models/churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to 'models/churn_model.pkl'.")