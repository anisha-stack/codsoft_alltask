import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv(r"N:\steph\archive (3)\creditcard.csv")
print(df.head())
print(df['Class'].value_counts())

# Normalize
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

# Handle imbalance
X = df.drop('Class', axis=1)
y = df['Class']
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(y_res.value_counts())
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Fraud Detection Confusion Matrix')
plt.show()
import joblib
joblib.dump(rf_model, 'fraud_detection_model.pkl')
model = joblib.load('fraud_detection_model.pkl')
new_df = pd.read_csv("new_transactions.csv")
new_df['Amount'] = scaler.transform(new_df[['Amount']])
new_df['Time'] = scaler.transform(new_df[['Time']])
predictions = rf_model.predict(new_df)
print(predictions)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf_model, X_res, y_res, cv=5)
print("Average accuracy:", scores.mean())
import pandas as pd
import matplotlib.pyplot as plt

importances = rf_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
plt.gca().invert_yaxis()
plt.title('Top 15 Important Features in Fraud Detection')
plt.xlabel('Importance')
plt.show()
report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).to_csv('fraud_detection_report.csv')
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='f1')
grid_search.fit(X_res, y_res)
print("Best parameters:", grid_search.best_params_)
