import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
#from sklearn.preprocessing import StandardScaler
#from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Load data
df = pd.read_excel('/Users/gabrielerizzo/Downloads/Aigab/dativ1.5.xlsx', sheet_name='TRAIN')

# Separate features and target
X = df.drop(['RACCOMANDAZIONE'], axis=1)
y = df['RACCOMANDAZIONE']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = ImbPipeline(steps=[
    ('oversampling', SMOTE(random_state=42)),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Definizione della griglia di parametri per GridSearchCV
param_grid = {
    'classifier__n_estimators': [1, 20000],
    'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3, 0.5],
    'classifier__max_depth': [3, 5, 7, 10, 15]
}

# Setup della cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Ricerca grid con cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

# Modello migliore
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Funzione per salvare il modello
def save_model(model, save_path='models/financial_product_advisor.joblib'):
    joblib.dump(model, save_path)
    print(f"Model saved at {save_path}.")

# Salvataggio interattivo del modello
if input("Do you want to save the model? (yes/no): ").lower() in ['yes', 'y']:
    save_model(best_model)
else:
    print("Model not saved.")