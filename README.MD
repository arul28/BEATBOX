# BEATBOX - Drum Sound Classification System

A machine learning system for classifying drum sounds using a hybrid KNN approach with personalization capabilities.

## Model Comparison

| Approach      | Description                                                                 | Accuracy / ARI | Strengths                              | Limitations                                   |
|---------------|-----------------------------------------------------------------------------|----------------|----------------------------------------|-----------------------------------------------|
| Unsupervised  | Clustering with K-Means, GMM, HDBSCAN, Hierarchical                         | ARI ≈ 0.11     | No labels needed, flexible discovery   | Poor label alignment, low structure clarity   |
| Supervised    | Global KNN model trained on all users, tested on unseen participants        | 57.0% Accuracy | Simple, interpretable                  | Weak generalization to new users              |
| Hybrid        | KNN with 5 user-specific samples + feature weighting via Ridge/Lasso        | 83.8% Accuracy | Personalized, low data requirement     | Requires minimal user input at runtime        |


## Dataset

This project uses the combined **AVP-LVT Vocal Percussion Dataset**, which merges:
- AVP (Amateur Vocal Percussion) — 28 performers
- LVT (Live Vocalised Transcription) — 20 performers

Both datasets are preprocessed and included in the repository under the `data/` directory. These include segmented audio clips, standardized labels, and annotation metadata for training, evaluation, and synthesis.


## Setup Instructions

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
.\venv\Scripts\activate  # On Windows
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Workflow

1. **Data Collection and Processing**
   - Run `datacollection.ipynb` to create a master dataframe of all audio samples
   - Run `audiosegmentation.ipynb` to segment audio into individual drum sounds
   - Run `segmentaugmentation.ipynb` to create augmented segments for training

2. **Feature Extraction**
   - Navigate to the `feature_extraction` folder
   - Run all three feature extraction notebooks:
     - `mfcc_extraction.ipynb`
     - `envelope_features.ipynb`
     - `feature_selection.ipynb`

3. **Model Training**
   - Navigate to the `models` folder
   - Run the model files

4. **Final Pipeline**
   - Navigate to `final_pipeline/code/`
   - Run `pipeline.ipynb` to execute the complete system

## Key Features

- Hybrid KNN model with personalization capabilities
- Feature extraction using MFCCs and envelope features
- Data augmentation for improved model robustness
- Ridge and Lasso regression for feature selection
- Visualization tools for model analysis

## Model Architecture

The system uses a hybrid KNN approach that:
- Combines base training data with user-specific examples
- Uses Ridge/Lasso regression to identify important features
- Implements weighted distance metrics for improved classification
- Adapts to individual playing styles
