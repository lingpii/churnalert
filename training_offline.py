import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ... (Copy toàn bộ logic tạo dataframe và pipeline của bạn vào đây) ...

pipeline.fit(X_tr, y_tr)
calibrated = CalibratedClassifierCV(estimator=pipeline, method="isotonic", cv="prefit")
calibrated.fit(X_val, y_val)

# Tính toán các metrics và threshold
y_prob = calibrated.predict_proba(X_test)[:, 1]
prec, rec, thresh = precision_recall_curve(y_test, y_prob)
f1s = 2*prec*rec/(prec+rec+1e-9)
best_thresh = thresh[np.argmax(f1s[:-1])]

meta_data = {
    "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    "pr_auc":  round(average_precision_score(y_test, y_prob), 4),
    "f1":      round(f1_score(y_test, y_prob >= best_thresh), 4),
    "threshold": round(float(best_thresh), 3),
}

# 🚨 LƯU MODEL VÀ METRICS LẠI
joblib.dump({"model": calibrated, "meta": meta_data}, "churn_model_v1.pkl")
print("Đã lưu model thành công!")