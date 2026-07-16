# Phishing URL Detector

AI-powered system that detects phishing URLs using machine learning, built with Python.

## Overview

This project uses a Random Forest classifier to identify malicious/phishing URLs, trained on a dataset of 58,000+ URLs.

## Results

- **Accuracy:** 95.8%
- **Precision & Recall:** 0.96 (balanced across both phishing and legitimate URL classes)
- Only 490 misclassifications out of 11,729 test samples

## Tech Stack

- Python
- Scikit-learn (Random Forest)
- Pandas for data processing
- Matplotlib for visualization

## Files

- `phishing_detector.py` — Main model training and evaluation script
- `explore_data.py` — Data exploration and preprocessing
- `confusion_matrix.png` — Model performance visualization

## Limitations

- Currently misses ~3.7% of phishing URLs (false negatives)
- Requires pre-extracted features; not yet tested on raw live URLs

## Future Work

- Live URL feature extraction for real-time predictions
- Flask web interface for interactive URL checking
- Hyperparameter tuning to improve recall on phishing class

## Author

Aroon Kumar Maheshwari
