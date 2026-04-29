import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import joblib

def main():
    # 1. Dataset
    print("Loading dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    df = pd.concat([X, y], axis=1)

    print("First few rows:")
    print(df.head())
    print("\nShape:", df.shape)

    # 2. EDA
    print("\nPerforming EDA...")
    os.makedirs('outputs', exist_ok=True)

    eda_report = []
    eda_report.append("=== EDA ===")
    eda_report.append(f"Missing values:\n{df.isnull().sum().sum()} total missing values")
    eda_report.append(f"Duplicates: {df.duplicated().sum()}")
    eda_report.append(f"Data Types:\n{df.dtypes.value_counts().to_string()}")
    eda_report.append(f"Summary Statistics:\n{df.describe().to_string()}")

    # Plot target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df)
    plt.title('Class Balance')
    plt.savefig('outputs/class_balance.png')
    plt.close()

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()

    # Plot distribution of mean radius (example feature)
    plt.figure(figsize=(6, 4))
    sns.histplot(df['mean radius'], kde=True)
    plt.title('Distribution of Mean Radius')
    plt.savefig('outputs/feature_distribution.png')
    plt.close()

    # 3. Data Preprocessing
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 4. Model Training
    print("Training models...")

    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=10000, random_state=42),
            'params': {'C': [0.1, 1.0, 10.0]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
        },
        'XGBoost': {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'params': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
        }
    }

    best_models = {}
    for name, m in models.items():
        print(f"Tuning {name}...")
        clf = GridSearchCV(m['model'], m['params'], cv=5, scoring='accuracy', n_jobs=-1)
        clf.fit(X_train_processed, y_train)
        best_models[name] = clf.best_estimator_
        print(f"Best params for {name}: {clf.best_params_}")

    # 5. Model Evaluation
    print("Evaluating models...")
    results = []
    for name, model in best_models.items():
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })

    results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
    print("\nModel Comparison:")
    print(results_df.to_string())

    best_model_name = results_df.iloc[0]['Model']
    best_model = best_models[best_model_name]
    print(f"\nBest Model: {best_model_name}")

    # Plot Confusion Matrix for Best Model
    y_pred_best = best_model.predict(X_test_processed)
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.savefig('outputs/best_model_confusion_matrix.png')
    plt.close()

    # Plot ROC Curve for Best Model
    y_prob_best = best_model.predict_proba(X_test_processed)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob_best)
    roc_auc_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {best_model_name}')
    plt.legend(loc="lower right")
    plt.savefig('outputs/best_model_roc_curve.png')
    plt.close()

    # 6. Save Everything
    print("Saving artifacts...")
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')

    with open('ml_report.txt', 'w') as f:
        f.write("Machine Learning Project Report\n")
        f.write("===============================\n\n")
        f.write("\n".join(eda_report))
        f.write("\n\n=== Model Evaluation ===\n")
        f.write(results_df.to_string())
        f.write(f"\n\nBest Model selected: {best_model_name}\n")
        f.write(f"Reason: Highest F1-score ({results_df.iloc[0]['F1-Score']:.4f}) and ROC-AUC ({results_df.iloc[0]['ROC-AUC']:.4f})\n")
        
    print("Project completed successfully.")

if __name__ == "__main__":
    main()
