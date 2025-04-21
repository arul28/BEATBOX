import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data():
    """Load embeddings and labels"""
    base_dir = Path('cnn')
    
    # Load all data
    embeddings = np.load('C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/code/models/cnn/embeddings.npy')
    instrument_labels = np.load('C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/code/models/cnn/instrument_labels.npy')
    
    return embeddings, instrument_labels

def train_evaluate_knn(X_train, X_test, y_train, y_test):
    """Train and evaluate KNN classifier"""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize classifier with paper's parameters
    clf = KNeighborsClassifier(n_neighbors=5,
                              weights='distance',
                              metric='manhattan')
    
    # Train model
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = clf.predict(X_test_scaled)
    
    # Print results
    print("\nTest Set Results:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['hhc', 'hho', 'kd', 'sd']))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['hhc', 'hho', 'kd', 'sd'],
                yticklabels=['hhc', 'hho', 'kd', 'sd'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return clf, scaler

def main():
    print("Loading data...")
    embeddings, labels = load_data()
    
    print("\nSplitting data into 80% train and 20% test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print("\nLabel distribution in training set:")
    print(pd.Series(y_train).value_counts())
    print("\nLabel distribution in test set:")
    print(pd.Series(y_test).value_counts())
    
    print("\nTraining and evaluating KNN classifier...")
    clf, scaler = train_evaluate_knn(X_train, X_test, y_train, y_test)
    
    # Save the trained model and scaler
    import joblib
    joblib.dump(clf, 'knn_model.joblib')
    joblib.dump(scaler, 'knn_scaler.joblib')
    print("\nSaved model and scaler to 'knn_model.joblib' and 'knn_scaler.joblib'")

if __name__ == "__main__":
    main() 