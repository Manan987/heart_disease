import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Handle missing values
    data = data.replace('?', np.nan)
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    
    # Fill missing values with median for numerical columns
    data = data.fillna(data.median())
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    return X, y

# Feature importance analysis
def analyze_feature_importance(model, X, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importances:")
    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

# Model evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data('heart_disease_data.csv')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fix pipeline configuration
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('selector', SelectFromModel(RandomForestClassifier(random_state=42))),  # Match classifier type
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Update param grid for selector
    param_grid = {
        'selector__max_features': [10, 13],  # Control number of features
        'selector__estimator__n_estimators': [50, 100],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    
    # Define hyperparameters to search
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2']
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    print("Training model with grid search...")
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters:", grid_search.best_params_)
    
    # Evaluate the best model
    best_model = grid_search.best_estimator_
    evaluate_model(best_model, X_test, y_test)
    
    # Analyze feature importance
    final_features = best_model.named_steps['feature_selection'].get_support()
    selected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    print("\nSelected features:", selected_features)
    
    # Cross-validation score
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='f1')
    print(f"\nCross-validation F1 scores: {cv_scores}")
    print(f"Average CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save the model
    print("Saving model...")
    joblib.dump(best_model, 'model/heart_disease_model.pkl')
    print("Model saved successfully")
    
    return best_model

if __name__ == "__main__":
    main()