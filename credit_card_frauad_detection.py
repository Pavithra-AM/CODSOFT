import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

df = pd.read_csv('/content/creditcard.csv', on_bad_lines='skip')

print(df.info())
print(df.describe())

print(df.isnull().sum())

df = df.dropna(subset=['Class'])

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(df.mean())

plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.show()

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

fraud_indices = np.where(y_train == 1)[0]
non_fraud_indices = np.where(y_train == 0)[0]

random_non_fraud_indices = np.random.choice(non_fraud_indices, len(fraud_indices), replace=False)
undersample_indices = np.concatenate([fraud_indices, random_non_fraud_indices])

X_train_undersample = X_train[undersample_indices]
y_train_undersample = y_train.iloc[undersample_indices]

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_undersample, y_train_undersample)

y_pred_lr = lr_model.predict(X_test)

print("Logistic Regression Model Performance:")
print("Precision: ", precision_score(y_test, y_pred_lr))
print("Recall: ", recall_score(y_test, y_pred_lr))
print("F1-Score: ", f1_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_undersample, y_train_undersample)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Model Performance:")
print("Precision: ", precision_score(y_test, y_pred_rf))
print("Recall: ", recall_score(y_test, y_pred_rf))
print("F1-Score: ", f1_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()
