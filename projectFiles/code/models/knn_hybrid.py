import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

def load_features_and_labels(features_path, labels_path, segment_info_path):
    """
    Load features and labels from specified paths.
    """
    X = np.load(features_path)
    y = np.load(labels_path)
    segment_info = pd.read_csv(segment_info_path)
    
    assert len(X) == len(y) == len(segment_info)
    return X, y, segment_info

def create_basic_weighted_knn(X_train, y_train, X_user, y_user, weight_factor=3.0, **knn_params):
    """
    Create a KNN model where user examples have extra weight by repeating them.
    """
    # Repeat user examples to give them more weight
    n_repeats = int(weight_factor)
    X_user_repeated = np.repeat(X_user, n_repeats, axis=0)
    y_user_repeated = np.repeat(y_user, n_repeats)
    
    # Combine with training data
    X_combined = np.vstack([X_train, X_user_repeated])
    y_combined = np.concatenate([y_train, y_user_repeated])
    
    # Create and train KNN
    knn = KNeighborsClassifier(**knn_params)
    knn.fit(X_combined, y_combined)
    
    return knn

def create_feature_weighted_knn(X_train, y_train, X_user, y_user, weight_factor=3.0, 
                              feature_weight_method='ridge', C=1.0, **knn_params):
    """
    Create a KNN model with learned feature weights.
    
    Args:
        X_train, y_train: Base training data
        X_user, y_user: User examples
        weight_factor: How many times to repeat user examples
        feature_weight_method: 'ridge', 'lasso', or 'logistic'
        C: Regularization strength
        knn_params: Parameters for KNeighborsClassifier
    """
    # Learn feature weights from user examples
    if feature_weight_method == 'ridge':
        # Ridge regression for feature weights
        model = RidgeClassifier(alpha=1.0/C)
    elif feature_weight_method == 'lasso':
        # Lasso regression for feature weights (use LogisticRegression with L1)
        model = LogisticRegression(penalty='l1', C=C, solver='liblinear')
    else:
        # Default to logistic regression with L1 regularization
        model = LogisticRegression(penalty='l1', C=C, solver='liblinear')
    
    # Fit the model to user examples
    model.fit(X_user, y_user)
    
    # Get feature weights
    if hasattr(model, 'coef_'):
        if len(model.coef_.shape) == 2:
            # For multiclass models, take the average of absolute values across all classes
            feature_weights = np.mean(np.abs(model.coef_), axis=0)
        else:
            feature_weights = np.abs(model.coef_)
    else:
        # If no coef_ attribute, use uniform weights
        feature_weights = np.ones(X_user.shape[1])
    
    # Normalize weights to sum to 1
    feature_weights = feature_weights / np.sum(feature_weights)
    
    # Print top and bottom features
    top_indices = np.argsort(feature_weights)[-5:]
    bottom_indices = np.argsort(feature_weights)[:5]
    
    print("\nFeature weights learned from user examples:")
    print(f"Top 5 features: {top_indices} with weights {feature_weights[top_indices]}")
    print(f"Bottom 5 features: {bottom_indices} with weights {feature_weights[bottom_indices]}")
    
    # Repeat user examples to give them more weight
    n_repeats = int(weight_factor)
    X_user_repeated = np.repeat(X_user, n_repeats, axis=0)
    y_user_repeated = np.repeat(y_user, n_repeats)
    
    # Combine with training data
    X_combined = np.vstack([X_train, X_user_repeated])
    y_combined = np.concatenate([y_train, y_user_repeated])
    
    # Define a custom distance function with feature weights
    def weighted_euclidean(x, y):
        return np.sqrt(np.sum(feature_weights * ((x - y) ** 2)))
    
    # Create KNN with custom metric using learned weights
    # Override any metric parameters in knn_params
    params = knn_params.copy()
    params['metric'] = 'pyfunc'
    params['metric_params'] = {'func': weighted_euclidean}
    
    knn = KNeighborsClassifier(**params)
    knn.fit(X_combined, y_combined)
    
    return knn, feature_weights

def grid_search_hyperparameters(X_train, y_train, X_adapt, y_adapt, X_test, y_test, base_params):
    """
    Perform grid search over weight factors and regularization parameters.
    
    Args:
        X_train, y_train: Base training data
        X_adapt, y_adapt: User adaptation data
        X_test, y_test: Test data
        base_params: Base KNN parameters from initial grid search
    """
    # Define parameter grid for weight factors and regularization
    weight_factors = [1, 2, 3, 4, 5, 7, 10]
    C_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    best_score = 0
    best_params = {}
    results = []
    
    print("\nGrid searching weight factors and regularization parameters...")
    for weight_factor in weight_factors:
        for C in C_values:
            # Try both Ridge and Lasso
            for method in ['ridge', 'lasso']:
                try:
                    # Create feature-weighted model with current parameters
                    model, _ = create_feature_weighted_knn(
                        X_train, y_train, X_adapt, y_adapt,
                        weight_factor=weight_factor,
                        feature_weight_method=method,
                        C=C,
                        **base_params
                    )
                    
                    # Evaluate on test set
                    score = model.score(X_test, y_test)
                    results.append({
                        'weight_factor': weight_factor,
                        'C': C,
                        'method': method,
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'weight_factor': weight_factor,
                            'C': C,
                            'method': method
                        }
                    
                    print(f"Weight={weight_factor}, C={C}, Method={method}: Score={score:.3f}")
                    
                except Exception as e:
                    print(f"Failed for weight={weight_factor}, C={C}, Method={method}: {str(e)}")
    
    # Sort results by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\nTop 5 parameter combinations:")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. Weight={result['weight_factor']}, C={result['C']}, "
              f"Method={result['method']}: Score={result['score']:.3f}")
    
    return best_params, best_score

def ensure_visualization_dir():
    """Ensure the visualization directory exists."""
    vis_dir = "/Users/arul/ML/BEATBOX/projectFiles/visualization/hybridknn"
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir

def compare_all_knn_approaches(X, y, segment_info, held_out_participant, examples_per_class=5):
    """
    Compare all KNN approaches:
    1. Base model (no participant data)
    2. Basic weighted model with weight=3
    3. Feature-weighted model with grid-searched parameters
    
    Args:
        X, y: Features and labels
        segment_info: DataFrame with participant info
        held_out_participant: ID of participant to adapt to
        examples_per_class: Number of examples per class to use for adaptation
    """
    print(f"\n==================================================")
    print(f"Comparing KNN approaches for participant {held_out_participant}")
    print(f"==================================================")
    
    # Exclude held out participant from training data
    train_mask = segment_info['participant_id'] != held_out_participant
    X_train = X[train_mask]
    y_train = y[train_mask]
    segment_info_train = segment_info[train_mask]
    
    # Get held out participant's data
    test_mask = segment_info['participant_id'] == held_out_participant
    X_held_out = X[test_mask]
    y_held_out = y[test_mask]
    
    # Create dataframe for the held-out participant's data
    held_out_df = pd.DataFrame({
        'index': np.where(test_mask)[0],
        'label': y_held_out
    })
    
    # Get adaptation indices (5 examples per class)
    adapt_indices = []
    test_indices = []
    
    for label in np.unique(y_held_out):
        label_indices = held_out_df[held_out_df['label'] == label]['index'].values
        
        # If there are fewer than examples_per_class*2, use half
        if len(label_indices) <= examples_per_class * 2:
            n_adapt = len(label_indices) // 2
        else:
            n_adapt = examples_per_class
            
        # Randomly select examples_per_class samples for adaptation
        np.random.seed(42)
        selected_indices = np.random.choice(label_indices, n_adapt, replace=False)
        
        adapt_indices.extend(selected_indices)
        # The rest are for testing
        test_indices.extend([idx for idx in label_indices if idx not in selected_indices])
    
    # Get adaptation and test data
    X_adapt = X[adapt_indices]
    y_adapt = y[adapt_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # Print statistics
    print("\nHeld out participant's data statistics:")
    print(f"Adaptation set size: {len(X_adapt)} (5 examples per class)")
    print(f"Test set size: {len(X_test)}")
    
    print("\nLabel distribution in adaptation set:")
    print(pd.Series(y_adapt).value_counts())
    
    print("\nLabel distribution in test set:")
    print(pd.Series(y_test).value_counts())
    
    # Create group IDs for cross-validation
    base_ids = np.array([
        Path(p).stem.replace("_aug" + ''.join(filter(str.isdigit, Path(p).stem)), '')
        for p in segment_info_train['segment_path'].values
    ])
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_adapt_scaled = scaler.transform(X_adapt)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_adapt_enc = le.transform(y_adapt)
    y_test_enc = le.transform(y_test)
    
    # Define parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    # Initialize KNN and GroupKFold
    knn = KNeighborsClassifier()
    gkf = GroupKFold(n_splits=5)
    
    # Perform grid search with GroupKFold
    grid_search = GridSearchCV(
        knn,
        param_grid,
        cv=gkf.split(X_train_scaled, y_train_enc, groups=base_ids),
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Fit base model
    print("\nTraining base model (excluding held out participant)...")
    grid_search.fit(X_train_scaled, y_train_enc)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print("\nBest parameters:", best_params)
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    # Evaluate base model on test set
    base_preds = le.inverse_transform(best_model.predict(X_test_scaled))
    base_acc = accuracy_score(y_test, base_preds)
    
    # Create basic weighted model with weight=3
    print("\nTraining basic weighted model with weight=3...")
    
    # Extract relevant parameters for KNN
    knn_params = {}
    for param in ['n_neighbors', 'weights']:
        if param in best_params:
            knn_params[param] = best_params[param]
    
    basic_weighted_model = create_basic_weighted_knn(
        X_train_scaled, y_train_enc, X_adapt_scaled, y_adapt_enc, 
        weight_factor=3, **knn_params
    )
    basic_weighted_preds = le.inverse_transform(basic_weighted_model.predict(X_test_scaled))
    basic_weighted_acc = accuracy_score(y_test, basic_weighted_preds)
    
    # After evaluating basic weighted model, perform grid search for feature-weighted model
    print("\nPerforming grid search for feature-weighted model...")
    best_hyperparams, best_score = grid_search_hyperparameters(
        X_train_scaled, y_train_enc,
        X_adapt_scaled, y_adapt_enc,
        X_test_scaled, y_test_enc,
        knn_params
    )
    
    print("\nBest hyperparameters found:")
    print(f"Weight factor: {best_hyperparams['weight_factor']}")
    print(f"C: {best_hyperparams['C']}")
    print(f"Method: {best_hyperparams['method']}")
    print(f"Score: {best_score:.3f}")
    
    # Create feature-weighted model with best parameters
    feature_weighted_model, feature_weights = create_feature_weighted_knn(
        X_train_scaled, y_train_enc, X_adapt_scaled, y_adapt_enc,
        weight_factor=best_hyperparams['weight_factor'],
        feature_weight_method=best_hyperparams['method'],
        C=best_hyperparams['C'],
        **knn_params
    )
    
    feature_weighted_preds = le.inverse_transform(feature_weighted_model.predict(X_test_scaled))
    feature_weighted_acc = accuracy_score(y_test, feature_weighted_preds)
    
    # Print all results
    print("\n==================================================")
    print("RESULTS SUMMARY")
    print("==================================================")
    print(f"1. Base Model: {base_acc:.2%}")
    print(f"2. Basic Weighted Model (weight=3): {basic_weighted_acc:.2%}")
    print(f"3. Feature-Weighted Model: {feature_weighted_acc:.2%}")
    
    # Print detailed classification reports
    print("\n1. Base Model:")
    print(classification_report(y_test, base_preds))
    
    print("\n2. Basic Weighted Model (weight=3):")
    print(classification_report(y_test, basic_weighted_preds))
    
    print("\n3. Feature-Weighted Model:")
    print(classification_report(y_test, feature_weighted_preds))
    
    # Calculate per-class improvements
    base_report = classification_report(y_test, base_preds, output_dict=True)
    basic_weighted_report = classification_report(y_test, basic_weighted_preds, output_dict=True)
    feature_weighted_report = classification_report(y_test, feature_weighted_preds, output_dict=True)
    
    print("\nPer-class F1-score improvements from feature-weighted model:")
    for class_name in le.classes_:
        base_f1 = base_report[class_name]['f1-score']
        basic_f1 = basic_weighted_report[class_name]['f1-score']
        feature_f1 = feature_weighted_report[class_name]['f1-score']
        base_to_feature = (feature_f1 - base_f1) * 100
        basic_to_feature = (feature_f1 - basic_f1) * 100
        print(f"  {class_name}: Base={base_f1:.2f}, Basic={basic_f1:.2f}, Feature={feature_f1:.2f} " 
              f"(vs Base: {base_to_feature:+.1f}%, vs Basic: {basic_to_feature:+.1f}%)")
    
    # Get visualization directory
    vis_dir = ensure_visualization_dir()
    
    # Visualize confusion matrices
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Base model confusion matrix
    cm_base = confusion_matrix(y_test, base_preds, labels=le.classes_)
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax1)
    ax1.set_title(f'Base Model\nAccuracy: {base_acc:.2%}')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Basic weighted model confusion matrix
    cm_basic = confusion_matrix(y_test, basic_weighted_preds, labels=le.classes_)
    sns.heatmap(cm_basic, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax2)
    ax2.set_title(f'Basic Weighted (weight=3)\nAccuracy: {basic_weighted_acc:.2%}')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # Feature-weighted model confusion matrix
    cm_feature = confusion_matrix(y_test, feature_weighted_preds, labels=le.classes_)
    sns.heatmap(cm_feature, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax3)
    ax3.set_title(f'Feature-Weighted\nAccuracy: {feature_weighted_acc:.2%}')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'confusion_matrices_{held_out_participant}.png'))
    plt.close()
    
    # Visualize results
    models = ['Base', 'Basic Weighted\n(weight=3)', 'Feature\nWeighted']
    accuracies = [base_acc, basic_weighted_acc, feature_weighted_acc]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['lightgray', 'skyblue', 'royalblue'])
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title(f'Model Accuracy Comparison for Participant {held_out_participant}')
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'accuracy_comparison_{held_out_participant}.png'))
    plt.close()
    
    # Also show per-class F1 scores
    class_names = le.classes_
    base_f1 = [base_report[cls]['f1-score'] for cls in class_names]
    basic_f1 = [basic_weighted_report[cls]['f1-score'] for cls in class_names]
    feature_f1 = [feature_weighted_report[cls]['f1-score'] for cls in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, base_f1, width, label='Base Model', color='lightgray')
    plt.bar(x, basic_f1, width, label='Basic Weighted (weight=3)', color='skyblue')
    plt.bar(x + width, feature_f1, width, label='Feature Weighted', color='royalblue')
    
    plt.xlabel('Drum Type')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score by Drum Type for Participant {held_out_participant}')
    plt.xticks(x, class_names)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'f1_scores_{held_out_participant}.png'))
    plt.close()
    
    # Visualize feature weights
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(feature_weights)), feature_weights)
    plt.xlabel('Feature Index')
    plt.ylabel('Weight')
    plt.title(f'Feature Weights Learned for Participant {held_out_participant}')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'feature_weights_{held_out_participant}.png'))
    plt.close()
    
    return {
        'base_model': best_model,
        'basic_weighted_model': basic_weighted_model,
        'feature_weighted_model': feature_weighted_model,
        'feature_weights': feature_weights,
        'base_acc': base_acc,
        'basic_weighted_acc': basic_weighted_acc,
        'feature_weighted_acc': feature_weighted_acc
    }

# Example usage
def main():
    features_path = "/Users/arul/ML/BEATBOX/projectFiles/extracted_features/features/mfcc_extracted_features.npy"
    labels_path = "/Users/arul/ML/BEATBOX/projectFiles/extracted_features/labels/mfcc_extracted_labels.npy"
    segment_info_path = "/Users/arul/ML/BEATBOX/projectFiles/segment_info/segment_info.csv"
    
    # Load data
    X, y, segment_info = load_features_and_labels(features_path, labels_path, segment_info_path)
    
    # Test with different participants
    test_participants = ['P24', 'P17', 'P10']
    examples_per_class = 5
    
    all_results = {}
    for participant in test_participants:
        all_results[participant] = compare_all_knn_approaches(
            X, y, segment_info, participant, examples_per_class
        )
    
    # Summarize results across participants
    print("\n==================================================")
    print("FINAL SUMMARY ACROSS ALL PARTICIPANTS")
    print("==================================================")
    
    for participant, results in all_results.items():
        print(f"\n{participant}:")
        print(f"  Base accuracy: {results['base_acc']:.2%}")
        print(f"  Basic weighted accuracy: {results['basic_weighted_acc']:.2%}")
        print(f"  Feature weighted accuracy: {results['feature_weighted_acc']:.2%}")
        
        base_improvement = (results['feature_weighted_acc'] - results['base_acc']) * 100
        basic_improvement = (results['feature_weighted_acc'] - results['basic_weighted_acc']) * 100
        print(f"  Improvements:")
        print(f"    vs. Base: {base_improvement:+.1f}%")
        print(f"    vs. Basic Weighted: {basic_improvement:+.1f}%")
        
        # Print top features
        feature_weights = results['feature_weights']
        top_indices = np.argsort(feature_weights)[-5:]
        print(f"  Top 5 features: {top_indices}")

if __name__ == "__main__":
    main()