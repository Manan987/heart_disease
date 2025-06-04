import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
import os

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    """Load and preprocess the heart disease dataset"""
    data = pd.read_csv(file_path)
    
    # Handle missing values
    data = data.replace('?', np.nan)
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    
    # Fill missing values with median for numerical columns
    data = data.fillna(data.median())
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Check if it's binary or multiclass
    unique_classes = y.unique()
    print(f"Target classes found: {sorted(unique_classes)}")
    print(f"Number of classes: {len(unique_classes)}")
    
    # For heart disease, often classes > 0 indicate presence of disease
    # Convert to binary if needed (0 = no disease, 1 = disease)
    if len(unique_classes) > 2:
        print("Converting to binary classification (0 = no disease, >0 = disease)")
        y = (y > 0).astype(int)
        print(f"New class distribution: {y.value_counts().sort_index()}")
    
    return X, y

# Feature importance analysis
def analyze_feature_importance(model, X, feature_names):
    """Analyze and display feature importance"""
    # Get the final classifier from the pipeline
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps['classifier']
    else:
        classifier = model
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        
        # If feature selection was used, map back to original features
        if hasattr(model, 'named_steps') and 'selector' in model.named_steps:
            selector = model.named_steps['selector']
            selected_mask = selector.get_support()
            selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        else:
            selected_features = feature_names
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        print("\nTop Feature Importances:")
        for i in range(min(10, len(importances))):  # Show top 10
            idx = indices[i]
            print(f"{i+1:2d}. {selected_features[idx]:15s} ({importances[idx]:.4f})")
    else:
        print("Model doesn't support feature importance analysis")

# Model evaluation with proper multiclass handling
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with appropriate metrics"""
    y_pred = model.predict(X_test)
    
    # Determine if binary or multiclass
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    is_binary = len(unique_classes) <= 2
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    if is_binary:
        # Binary classification metrics
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    else:
        # Multiclass classification metrics
        print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
        print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.4f}")
        print(f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
        print(f"F1 Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data('heart_disease_data.csv')
    
    # Define feature names (typical heart disease dataset features)
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Ensure we have the right number of feature names
    if len(feature_names) != X.shape[1]:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        print(f"Using generic feature names for {X.shape[1]} features")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Determine scoring method based on number of classes
    unique_classes = y.unique()
    is_binary = len(unique_classes) <= 2
    scoring_method = 'f1' if is_binary else 'f1_weighted'
    
    print(f"Using scoring method: {scoring_method}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('selector', SelectFromModel(RandomForestClassifier(random_state=42))),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define hyperparameters to search
    param_grid = {
        'selector__max_features': [10, 13] if X.shape[1] >= 13 else [X.shape[1]//2, X.shape[1]],
        'selector__estimator__n_estimators': [50, 100],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__max_features': ['sqrt', 'log2']
    }
    
    # Perform grid search with cross-validation
    print("Performing grid search with cross-validation...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring=scoring_method,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")
    print("Best parameters:", grid_search.best_params_)
    
    # Evaluate the best model
    best_model = grid_search.best_estimator_
    evaluate_model(best_model, X_test, y_test)
    
    # Analyze feature importance
    analyze_feature_importance(best_model, X, feature_names)
    
    # Cross-validation score on full dataset
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring=scoring_method)
    print(f"\nCross-validation {scoring_method} scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save the model
    print("\nSaving model...")
    joblib.dump(best_model, 'model/heart_disease_model.pkl')
    
    # Save feature names for later use
    joblib.dump(feature_names, 'model/feature_names.pkl')
    
    # Save preprocessing info
    model_info = {
        'is_binary': is_binary,
        'classes': list(unique_classes),
        'feature_names': feature_names,
        'scoring_method': scoring_method
    }
    joblib.dump(model_info, 'model/model_info.pkl')
    
    print("Model and metadata saved successfully!")
    
    return best_model

if __name__ == "__main__":
    main()