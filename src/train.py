import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Load the datasets
try:
    fraud_df = pd.read_csv('../data/processed/cleaned_fraud_data.csv')
    credit_df = pd.read_csv('../data/processed/cleaned_credit_data.csv')
    print("Loaded Fraud and creadit data successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please check the file paths.")

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a classification model's performance on test data.

    Prints the confusion matrix, classification report, F1 score, and area under the precision-recall curve (AUC-PR).
    
    Parameters:
        model: Trained classification model with predict and predict_proba methods.
        X_test: Test features.
        y_test: True labels for test data.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)

    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")


# Prepare fraud detection data
# Assuming 'class' is the target variable in fraud_df
Xf = fraud_df.drop(columns=['class'])
yf = fraud_df['class']

print("Started one-hot encoding for fraud detection data...")

# One-hot encode before sampling
Xf = pd.get_dummies(Xf, columns=['source','browser','sex','country'], drop_first=True)


print("Started splitting fraud detection data...")
Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, stratify=yf, test_size=0.3, random_state=42)

print("Started SMOTE for fraud detection data...")
smote = SMOTE()
Xf_train_resampled, yf_train_resampled = smote.fit_resample(Xf_train, yf_train)

print("Started scaling for fraud detection data...")
scaler = StandardScaler()
# Only scale 'purchase_value'
Xf_train_resampled['purchase_value'] = scaler.fit_transform(Xf_train_resampled[['purchase_value']])
print("Transforming test data...")
Xf_test['purchase_value'] = scaler.transform(Xf_test[['purchase_value']])


# Prepare credit card fraud data
# Assuming 'Class' is the target variable in credit_df
Xc = credit_df.drop(columns='Class')
yc = credit_df['Class']


Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, stratify=yc, test_size=0.3, random_state=42)

# smote for credit card fraud data
smote = SMOTE()
Xc_train_resampled, yc_train_resampled = smote.fit_resample(Xc_train, yc_train)

scaler = StandardScaler()
# Only scale 'Amount'
Xc_train_resampled['Amount'] = scaler.fit_transform(Xc_train_resampled[['Amount']])
Xc_test['Amount'] = scaler.transform(Xc_test[['Amount']])

# Train Logistic Regression models
# Using max_iter=1000 to ensure convergence
log_fraud = LogisticRegression(max_iter=1000, random_state=42)
log_fraud.fit(Xf_train_resampled, yf_train_resampled)

log_credit = LogisticRegression(max_iter=1000, random_state=42)
log_credit.fit(Xc_train_resampled, yc_train_resampled)

# Train XGBoost models with appropriate scale_pos_weight
# Adjust scale_pos_weight based on the class imbalance
# For fraud detection, we assume a lower imbalance, hence a lower scale_pos_weight
xgb_fraud = XGBClassifier(scale_pos_weight=10, use_label_encoder=False, eval_metric='logloss')
xgb_fraud.fit(Xf_train_resampled, yf_train_resampled)

xgb_credit = XGBClassifier(scale_pos_weight=50, use_label_encoder=False, eval_metric='logloss')
xgb_credit.fit(Xc_train_resampled, yc_train_resampled)

# Evaluate models on test data
evaluate_model(log_fraud, Xf_test, yf_test)
evaluate_model(xgb_fraud, Xf_test, yf_test)

evaluate_model(log_credit, Xc_test, yc_test)
evaluate_model(xgb_credit, Xc_test, yc_test)
