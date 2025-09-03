import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('C:\\Users\\Lenovo\Desktop\\ML Internship\\Telco_Customer_Churn_Dataset.csv')
df.columns = df.columns.str.strip()
print("shape:", df.shape)
print(df.head())
print(df.info())
print("Churn value counts:")
print(df["Churn"].value_counts(dropna=False))
for id_col in ["customerID", "CustomerID", "customer id", "id"]:
    if id_col in df.columns:
        df = df.drop(columns=[id_col])
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
df["Churn"] = df["Churn"].astype(str).str.strip().str.lower().map({"yes":1, "no":0, "true":1, "false":0})
df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce")
df = df[df["Churn"].notna()].reset_index(drop=True)
df["Churn"] = df["Churn"].astype(int)
print("\nMissing values by column:")
print(df.isnull().sum().sort_values(ascending=False).head(30))
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in df.columns if c not in cat_cols and c != "Churn"]
print("\nNumerical columns:", num_cols)
print("Categorical columns:", cat_cols)
df.to_csv("telco_cleaned.csv", index=False)
print("Cleaned CSV saved: telco_cleaned.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, "Test:", X_test.shape)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
preferred = [
 "tenure","MonthlyCharges","TotalCharges","SeniorCitizen","Contract",
 "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
 "TechSupport","StreamingTV","StreamingMovies","PaymentMethod",
 "PaperlessBilling","Partner","Dependents","PhoneService","MultipleLines"
]
available = [c for c in preferred if c in X_train.columns]
if not available:
    available = [c for c in X_train.columns]
print("Using features:", available)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
numeric_features = [c for c in available if df[c].dtype != "O"]
categorical_features = [c for c in available if c not in numeric_features]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

pipelines = {name: Pipeline([("preproc", preprocessor), ("clf", model)]) for name, model in models.items()}
from sklearn.model_selection import cross_val_score
import numpy as np
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"{name} CV ROC-AUC mean: {np.mean(scores):.4f}  std: {np.std(scores):.4f}")
    pipe.fit(X_train, y_train)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import joblib

results = {}
for name, pipe in pipelines.items():
    y_pred = pipe.predict(X_test)
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:,1]
    else:
        try:
            raw = pipe.decision_function(X_test)
            raw = (raw - raw.min())/(raw.max()-raw.min()+1e-9)
            y_prob = raw
        except:
            y_prob = y_pred.astype(float)

    res = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "report": classification_report(y_test, y_pred, digits=3, zero_division=0)
    }
    results[name] = res
    print("=== ", name, " ===")
    for k,v in res.items():
        if k!="report": print(f"{k}: {v:.4f}")
    print(res["report"])
best_name = max(results, key=lambda k: results[k]["roc_auc"])
print("Best by ROC-AUC:", best_name, results[best_name]["roc_auc"])
fpr, tpr, _ = roc_curve(y_test, pipelines[best_name].predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC — {best_name}")
plt.show()
joblib.dump(pipelines[best_name], "best_telco_pipeline.pkl")
print("Saved best pipeline to best_telco_pipeline.pkl")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(8,6))

for name, pipe in pipelines.items():
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:,1]
    else:
        try:
            raw = pipe.decision_function(X_test)
            raw = (raw - raw.min())/(raw.max()-raw.min()+1e-9)
            y_prob = raw
        except:
            y_prob = pipe.predict(X_test).astype(float)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr,tpr):.3f})")

plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Model Comparison")
plt.legend()
plt.show()
df_results = pd.DataFrame(results).T[["accuracy","precision","recall","f1","roc_auc"]]
print(df_results)
df_results.plot(kind="bar", figsize=(10,6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.show()
from sklearn.metrics import ConfusionMatrixDisplay
best_model = pipelines[best_name]
y_pred = best_model.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title(f"Confusion Matrix – {best_name}")
plt.show()