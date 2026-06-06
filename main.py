import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data (Make sure the CSV is in your working directory)
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
df = pd.read_csv('creditcard.csv')

# 2. Basic Inspection
print("Dataset Shape:", df.shape)
print("Max Missing Values in any column:", df.isnull().sum().max()) 

# 3. Calculate the Class Imbalance 
class_counts = df['Class'].value_counts()
print("\n--- Class Distribution ---")
print(class_counts)
print(f"Percentage of Fraud: {(class_counts[1] / len(df)) * 100:.3f}%\n")

# 4. Visualize the Imbalance
plt.figure(figsize=(8, 6))
# We use a log scale because the difference is so massive; 
# on a linear scale, the fraud bar would be invisible.
sns.countplot(x='Class', data=df, hue='Class', palette='Set2', legend=False)
plt.title('Distribution of Normal (0) vs. Fraudulent (1) Transactions')
plt.yscale('log') 
plt.xlabel('Class (0 = Normal, 1 = Fraud)')
plt.ylabel('Count (Log Scale)')
plt.show()

# 5. Compare Transaction Amounts (Do fraudsters spend more?)
print("--- Normal Transaction Stats ---")
print(df[df['Class'] == 0]['Amount'].describe(), "\n")

print("--- Fraud Transaction Stats ---")
print(df[df['Class'] == 1]['Amount'].describe())

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 1. Separate Features (X) and Target (y)
# We drop the 'Class' column for our features, and keep it as our target
X = df.drop('Class', axis=1)
y = df['Class']

# 2. Perform a Stratified Train-Test Split
# Stratify ensures the rare 0.17% fraud rate is proportionally maintained in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("--- Before SMOTE ---")
print("Training set normal transactions:", sum(y_train == 0))
print("Training set fraud transactions:", sum(y_train == 1))

# 3. Apply SMOTE strictly to the Training Set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\n--- After SMOTE ---")
print("Training set normal transactions:", sum(y_train_smote == 0))
print("Training set fraud transactions:", sum(y_train_smote == 1))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Initialize the model 
# (n_jobs=-1 tells your computer to use all CPU cores to speed up training)
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

# 2. Train the model on the BALANCED SMOTE data
print("Training the Random Forest model... (This might take a minute)")
rf_model.fit(X_train_smote, y_train_smote)

# 3. Make predictions on the UNTOUCHED test data
print("Making predictions on the test set...")
y_pred = rf_model.predict(X_test)

# 4. Evaluate the results
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 5. Visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Fraud'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix on Test Data')
plt.show()

import shap
import matplotlib.pyplot as plt
import numpy as np

print("Initializing SHAP explainer... (This may take a moment)")

# 1. Initialize the explainer
explainer = shap.TreeExplainer(rf_model)

# 2. Sample the test data (500 rows for performance)
X_test_sample = X_test.sample(n=500, random_state=42)

# 3. Calculate SHAP values
shap_values = explainer.shap_values(X_test_sample)

# --- THE FIX ---
# Check if SHAP returned a list (legacy) or a 3D array (modern)
if isinstance(shap_values, list):
    fraud_shap_values = shap_values[1]
else:
    # Slice the 3D array: [All Rows, All Features, Class Index 1 (Fraud)]
    fraud_shap_values = shap_values[:, :, 1]

# 4. Generate the Global Feature Importance Plot
plt.figure(figsize=(10, 6))
plt.title("Global Feature Importance (What drives fraud overall?)")
shap.summary_plot(fraud_shap_values, X_test_sample, plot_type="bar", show=False)
plt.show()

# 5. Generate the Summary Dot Plot
plt.figure(figsize=(10, 6))
plt.title("SHAP Summary Plot (Detailed Impact)")
shap.summary_plot(fraud_shap_values, X_test_sample, show=False)
plt.show()

import joblib

print("Saving assets...")

# 1. Save the trained model and explainer
joblib.dump(rf_model, 'fraud_model.pkl')
joblib.dump(explainer, 'shap_explainer.pkl')

# 2. Save a tiny sample of our test data for the app to pull from
# We will grab 50 normal and 50 fraud cases
demo_data = pd.concat([
    X_test[y_test == 0].sample(50, random_state=42),
    X_test[y_test == 1].sample(50, random_state=42)
])
demo_labels = y_test.loc[demo_data.index]

demo_data.to_csv('demo_features.csv', index=False)
demo_labels.to_csv('demo_labels.csv', index=False)

print("Model and demo data saved successfully!")