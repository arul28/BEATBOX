import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data():
    """Load embeddings and metadata"""
    base_dir = Path('cnn')
    
    # Load all data
    embeddings = np.load('C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/code/models/cnn/embeddings.npy')
    instrument_labels = np.load('C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/code/models/cnn/instrument_labels.npy')
    participants = np.load('C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/code/models/cnn/participants.npy')
    
    # Load train/eval speaker split
    with open('C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/code/models/cnn/splits.json', 'r') as f:
        splits = json.load(f)
    
    return embeddings, instrument_labels, participants, splits

def create_train_test_split(embeddings, labels, participants, splits):
    """Split data into train (40 speakers) and test (8 speakers) sets"""
    # Create masks for train and test sets
    train_mask = np.isin(participants, splits['train_speakers'])
    test_mask = np.isin(participants, splits['eval_speakers'])
    
    # Split the data
    X_train = embeddings[train_mask]
    X_test = embeddings[test_mask]
    y_train = labels[train_mask]
    y_test = labels[test_mask]
    train_participants = participants[train_mask]
    test_participants = participants[test_mask]
    
    print(f"Training set: {X_train.shape[0]} samples from {len(splits['train_speakers'])} speakers")
    print(f"Test set: {X_test.shape[0]} samples from {len(splits['eval_speakers'])} speakers")
    
    return X_train, X_test, y_train, y_test, train_participants, test_participants

def evaluate_knn(X_train, X_test, y_train, y_test, train_participants):
    """Train and evaluate KNN classifier with 5-fold GroupKFold CV on training set"""
    # Scale features using only training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize classifier with paper's parameters
    clf = KNeighborsClassifier(n_neighbors=5,
                              weights='distance',
                              metric='manhattan')
    
    # Perform cross-validation on training set
    gkf = GroupKFold(n_splits=5)
    cv_scores = []
    
    print("\nCross-validation on training speakers:")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_scaled, y_train, groups=train_participants)):
        # Split data
        X_fold_train = X_train_scaled[train_idx]
        X_fold_val = X_train_scaled[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        # Train and evaluate
        clf.fit(X_fold_train, y_fold_train)
        score = clf.score(X_fold_val, y_fold_val)
        cv_scores.append(score)
        print(f"Fold {fold + 1}: {score:.3f}")
    
    print(f"Mean CV score: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
    
    # Train final model on all training data
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate on test set (8 held-out speakers)
    y_pred = clf.predict(X_test_scaled)
    
    # Print results
    print("\nTest Set Results (8 evaluation speakers):")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['hhc', 'hho', 'kd', 'sd']))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['hhc', 'hho', 'kd', 'sd'],
                yticklabels=['hhc', 'hho', 'kd', 'sd'])
    plt.title('Confusion Matrix (Evaluation Speakers)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return clf, scaler

def main():
    print("Loading data...")
    embeddings, labels, participants, splits = load_data()
    
    print("\nSplitting data into train (40 speakers) and test (8 speakers) sets...")
    X_train, X_test, y_train, y_test, train_participants, test_participants = \
        create_train_test_split(embeddings, labels, participants, splits)
    
    print("\nTraining and evaluating KNN classifier...")
    clf, scaler = evaluate_knn(X_train, X_test, y_train, y_test, train_participants)
    
    # Save the trained model and scaler
    import joblib
    joblib.dump(clf, 'knn_model.joblib')
    joblib.dump(scaler, 'knn_scaler.joblib')
    print("\nSaved model and scaler to 'knn_model.joblib' and 'knn_scaler.joblib'")

if __name__ == "__main__":
    main()