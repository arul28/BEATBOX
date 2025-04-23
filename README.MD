# BEATBOX - Drum Sound Classification System

A machine learning system for classifying drum sounds using a hybrid KNN approach with personalization capabilities.

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

## Project Structure

```
BEATBOX/
├── projectFiles/
│   ├── code/
│   │   ├── datacollection.ipynb
│   │   ├── audiosegmentation.ipynb
│   │   ├── segmentaugmentation.ipynb
│   │   ├── feature_extraction/
│   │   │   ├── mfcc_extraction.ipynb
│   │   │   ├── envelope_features.ipynb
│   │   │   └── feature_selection.ipynb
│   │   ├── models/
│   │   │   ├── knn_hybrid.py
│   │   │   └── [other model files]
│   │   └── final_pipeline/
│   │       └── code/
│   │           └── pipeline.ipynb
│   ├── audio/
│   ├── extracted_features/
│   └── visualization/
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
   - Run the model files, including `knn_hybrid.py`

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

## Requirements

See `requirements.txt` for a complete list of dependencies. Key packages include:
- numpy
- pandas
- scikit-learn
- librosa
- matplotlib
- seaborn

## Notes

- Ensure all audio files are in the correct format (WAV)
- Follow the workflow in order as each step depends on the previous one
- The system is designed to work with drum sounds but can be adapted for other audio classification tasks
