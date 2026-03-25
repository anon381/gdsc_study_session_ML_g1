import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def main():
    print("Loading data...")
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'rides.csv')
    plots_dir = os.path.join(base_dir, 'plots')
    models_dir = os.path.join(base_dir, 'models')
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    data = pd.read_csv(data_path)
    
    print("Preprocessing data...")
    X = data.drop(columns=["ride_price"])
    y_reg = data["ride_price"]
    
    # For classification, we define high cost as greater than the median price
    median_price = y_reg.median()
    y_clf = (y_reg > median_price).astype(int)
    
    numeric_features = ["distance_km", "duration_min"]
    categorical_features = ["time_of_day", "traffic_level", "weather", "demand_level", "pickup_zone"]
    
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])
    
    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )
    
    # ---------------------------------------------------------
    # 1. Linear Regression
    # ---------------------------------------------------------
    print("Training Linear Regression...")
    lr_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    lr_model.fit(X_train, y_reg_train)
    y_reg_pred_lr = lr_model.predict(X_test)
    
    rmse_lr = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_lr))
    r2_lr = r2_score(y_reg_test, y_reg_pred_lr)
    print(f"Linear Regression - RMSE: {rmse_lr:.2f}, R2: {r2_lr:.3f}")
    
    # Plot constraints setup
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1a. Plot LR Results
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_reg_test, y=y_reg_pred_lr, alpha=0.5, color='blue')
    lims = [min(y_reg_test.min(), y_reg_pred_lr.min()), max(y_reg_test.max(), y_reg_pred_lr.max())]
    plt.plot(lims, lims, "--", color="red", linewidth=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Linear Regression\nRMSE: {rmse_lr:.2f} | R²: {r2_lr:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'lr_actual_vs_predicted.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # 2. Random Forest Regressor
    # ---------------------------------------------------------
    print("Training Random Forest...")
    rf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1))
    ])
    rf_model.fit(X_train, y_reg_train)
    y_reg_pred_rf = rf_model.predict(X_test)
    
    rmse_rf = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_rf))
    r2_rf = r2_score(y_reg_test, y_reg_pred_rf)
    print(f"Random Forest - RMSE: {rmse_rf:.2f}, R2: {r2_rf:.3f}")
    
    # 2a. Plot RF Results
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_reg_test, y=y_reg_pred_rf, alpha=0.5, color='green')
    lims = [min(y_reg_test.min(), y_reg_pred_rf.min()), max(y_reg_test.max(), y_reg_pred_rf.max())]
    plt.plot(lims, lims, "--", color="red", linewidth=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Random Forest Regression\nRMSE: {rmse_rf:.2f} | R²: {r2_rf:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rf_actual_vs_predicted.png'))
    plt.close()
    
    # 2b. Feature Importance RF
    # Get feature names after one-hot encoding
    cat_encoder = rf_model.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_feature_names)
    
    importances = rf_model.named_steps["regressor"].feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rf_feature_importance.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # 3. Logistic Regression (Classification High vs Low Cost)
    # ---------------------------------------------------------
    print("Training Logistic Regression (Classification)...")
    clf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    clf_model.fit(X_train, y_clf_train)
    y_clf_pred = clf_model.predict(X_test)
    
    acc = accuracy_score(y_clf_test, y_clf_pred)
    print(f"Logistic Regression - Accuracy: {acc:.3f}")
    
    # 3a. Confusion Matrix
    cm = confusion_matrix(y_clf_test, y_clf_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Cost", "High Cost"])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    plt.title(f"Classification Confusion Matrix\nAccuracy: {acc:.3f}")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'clf_confusion_matrix.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # 4. Save Models
    # ---------------------------------------------------------
    print("Saving models...")
    joblib.dump(lr_model, os.path.join(models_dir, 'linear_regression.joblib'))
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.joblib'))
    joblib.dump(clf_model, os.path.join(models_dir, 'logistic_regression.joblib'))
    
    print("Pipeline complete. Models and plots saved.")

if __name__ == "__main__":
    main()
