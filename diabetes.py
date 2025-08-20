!pip install pandas numpy matplotlib seaborn scikit-learn xgboost flask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import zipfile
!pip install pandas numpy matplotlib seaborn scikit-learn xgboost flask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import zipfile
import os

import xgboost as xgb

df = pd.read_csv("/content/diabetes.csv")
print(df.head())  # Display first few rows
print(df.info())  # Check data types and missing values
print(df.isnull().sum())  # Ensure no missing values
print(df.describe())  # Statistical summary
df.fillna(df.mean(), inplace=True)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(columns=["Outcome"]))
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_scaled["Outcome"] = df["Outcome"]
X = df_scaled.drop(columns=["Outcome"])
y = df_scaled["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(penalty='l1', C=0.001, solver='liblinear', max_iter=50)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
rf_model = RandomForestClassifier(n_estimators=2, max_depth=1, min_samples_split=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators=5, learning_rate=0.5, max_depth=1, subsample=0.1, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))


xgb_model = xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=2, eval_metric="logloss")
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

from sklearn.ensemble import StackingClassifier
stacked_model = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=2))
    ],
    final_estimator=LogisticRegression()
)

stacked_model.fit(X_train, y_train)
y_pred_stacked = stacked_model.predict(X_test)
print("Stacked Model Accuracy:", accuracy_score(y_test, y_pred_stacked))