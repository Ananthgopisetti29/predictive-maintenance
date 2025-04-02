# Machine Failure Prediction Tool

## Overview
This tool is a GUI-based machine failure prediction system using machine learning. It allows users to load training and test data, train a model using a Random Forest classifier, and predict failures based on test data or dynamic inputs. The tool also provides model evaluation through confusion matrices, classification reports, feature importance analysis, and correlation heatmaps.

## Features
- **Load Training & Test Data**: Easily load CSV files for training and testing.
- **Train Model**: Utilizes a Random Forest classifier with SMOTE for data balancing.
- **Predict Failures**: Make predictions on test datasets and dynamic user inputs.
- **Model Evaluation**: Visualizes confusion matrices, classification reports, feature importance, and correlation heatmaps.
- **Reset Functionality**: Allows resetting of data and model.
- **Modern UI**: Built using Tkinter with a responsive and user-friendly interface.

## Installation
### Prerequisites
Ensure you have Python installed along with the required dependencies:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

## Usage
1. **Run the script**:
   ```bash
   python your_script.py
   ```
2. **Load training data** from a CSV file.
3. **Train the model** using the Random Forest classifier.
4. **Load test data** and predict failures.
5. **Evaluate model performance** with visualized reports.
6. **Make predictions** using dynamic input values.
7. **Reset** to clear the model and data.

## GUI Components
- **File Selectors**: Load training and test datasets.
- **Buttons**:
  - Load Training Data
  - Load Test Data
  - Train Model
  - Predict Failures (Test Data)
  - Predict from Dynamic Input
  - Reset
- **Result Visualizations**:
  - Confusion Matrix
  - Classification Report
  - Feature Importance Chart
  - Correlation Heatmap

## Model Details
- **Algorithm**: Random Forest Classifier
- **Data Balancing**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Metrics**: Accuracy, Precision, Recall, F1-Score

## Future Enhancements
- Implement additional machine learning models for comparison.
- Integrate real-time data streaming for predictive maintenance.
- Enhance GUI with better visualization and user experience improvements.

## License
This project is licensed under the MIT License.

---
**Author**: Ananth Lakshmi Kumar Gopisetti 🚀
