import pandas as pd
import numpy as np
from tkinter import filedialog, Tk, Label, Button, messagebox, Text, Toplevel, Entry, Frame
from tkinter.ttk import Notebook, Style
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import time
from threading import Thread

# Initialize the Tkinter GUI
root = Tk()
root.title("Machine Failure Prediction")
root.geometry("700x600")

# Apply modern styling
style = Style()
style.configure("TNotebook", tabposition='n')
style.configure("TButton", font=("Arial", 10), padding=6)
style.configure("TLabel", font=("Arial", 12))
style.configure("TEntry", padding=4)

# Global variables for data, model, and windows
train_data = None
test_data = None
model = None
dynamic_window = None  # For dynamic input predictions
test_data_window = None  # For test data predictions

# Load training data
def load_train_data():
    global train_data
    file_path = filedialog.askopenfilename(title="Select Training CSV File")
    try:
        train_data = pd.read_csv(file_path)
        messagebox.showinfo("Success", "Training data loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Error loading training data: {e}")

# Load test data
def load_test_data():
    global test_data
    file_path = filedialog.askopenfilename(title="Select Test CSV File")
    try:
        test_data = pd.read_csv(file_path)
        messagebox.showinfo("Success", "Test data loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Error loading test data: {e}")

# Train model function
def train_model():
    if train_data is None:
        messagebox.showerror("Error", "Please load training data first.")
        return

    # Prepare features and labels
    X = train_data.drop(columns=['machine_id', 'failure_label'])
    y = train_data['failure_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to balance data
    smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.value_counts().min() - 1))
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train Random Forest model
    global model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate model
    y_pred = model.predict(X_test)
    show_results(y_test, y_pred, X.columns)
    messagebox.showinfo("Success", "Model trained successfully!")

# Display model evaluation results
def show_results(y_true, y_pred, feature_columns):
    result_window = Toplevel(root)
    result_window.title("Model Evaluation Results")
    result_window.geometry("900x600")

    notebook = Notebook(result_window)

    # Confusion Matrix Tab
    cm_frame = Frame(notebook)
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"], ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("Actual Label")
    canvas_cm = FigureCanvasTkAgg(fig_cm, master=cm_frame)
    canvas_cm.draw()
    canvas_cm.get_tk_widget().pack()
    notebook.add(cm_frame, text="Confusion Matrix")

    # Classification Report Tab
    report_frame = Frame(notebook)
    report_text = Text(report_frame, width=70, height=15)
    report = classification_report(y_true, y_pred, target_names=["No Failure", "Failure"])
    report_text.insert("1.0", report)
    report_text.config(state="disabled")
    report_text.pack()
    notebook.add(report_frame, text="Classification Report")

    # Feature Importance Tab
    fi_frame = Frame(notebook)
    feature_importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
    feature_importance.plot(kind='bar', ax=ax_fi)
    ax_fi.set_title("Feature Importance")
    ax_fi.set_ylabel("Importance")
    canvas_fi = FigureCanvasTkAgg(fig_fi, master=fi_frame)
    canvas_fi.draw()
    canvas_fi.get_tk_widget().pack()
    notebook.add(fi_frame, text="Feature Importance")

    # Correlation Heatmap Tab
    corr_frame = Frame(notebook)
    corr = train_data.select_dtypes(include=[np.number]).corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
    ax_corr.set_title("Feature Correlation Heatmap")
    canvas_corr = FigureCanvasTkAgg(fig_corr, master=corr_frame)
    canvas_corr.draw()
    canvas_corr.get_tk_widget().pack()
    notebook.add(corr_frame, text="Correlation Heatmap")

    notebook.pack(expand=1, fill='both')

# Predict failures for test data
def predict_test_data():
    global test_data_window
    if model is None:
        messagebox.showerror("Error", "Please train the model first.")
        return

    if test_data is None:
        messagebox.showerror("Error", "Please load test data first.")
        return

    # Destroy any existing test data window
    if test_data_window and test_data_window.winfo_exists():
        test_data_window.destroy()

    try:
        # Ensure the test data has the same structure as the training data
        test_features = test_data.drop(columns=['machine_id', 'failure_label'], errors='ignore')
        predictions = model.predict(test_features)

        # Display results in a new window
        test_data_window = Toplevel(root)
        test_data_window.title("Test Data Predictions")
        test_data_window.geometry("600x400")
        result_text = Text(test_data_window, height=20, width=60)
        result_text.pack(pady=10)
        result_text.insert("1.0", "Predictions for Test Data (0 = No Failure, 1 = Failure):\n")
        for i, prediction in enumerate(predictions):
            result_text.insert("end", f"Machine {i+1}: {'Failure' if prediction == 1 else 'No Failure'}\n")
    except Exception as e:
        messagebox.showerror("Error", f"Error during prediction: {e}")

# Predict failures for dynamic inputs
def predict_dynamic_input():
    global dynamic_window
    if model is None:
        messagebox.showerror("Error", "Please train the model first.")
        return

    # Destroy any existing dynamic window
    if dynamic_window and dynamic_window.winfo_exists():
        dynamic_window.destroy()

    try:
        input_values = [float(entry.get()) for entry in dynamic_inputs]
        prediction = model.predict([input_values])[0]
        result = "Failure" if prediction == 1 else "No Failure"

        # Display result in a new window
        dynamic_window = Toplevel(root)
        dynamic_window.title("Dynamic Input Prediction")
        dynamic_window.geometry("400x200")
        Label(dynamic_window, text=f"Prediction: {result}", font=("Arial", 14)).pack(pady=20)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric inputs.")

# Reset data and model
def reset():
    global train_data, test_data, model, dynamic_window, test_data_window
    train_data = None
    test_data = None
    model = None
    if dynamic_window and dynamic_window.winfo_exists():
        dynamic_window.destroy()
    if test_data_window and test_data_window.winfo_exists():
        test_data_window.destroy()
    messagebox.showinfo("Reset", "Data, model, and windows have been reset.")

# GUI Layout
Label(root, text="Machine Failure Prediction Tool", font=("Arial", 16)).pack(pady=10)
Button(root, text="Load Training Data", command=load_train_data, width=20).pack(pady=5)
Button(root, text="Load Test Data", command=load_test_data, width=20).pack(pady=5)
Button(root, text="Train Model", command=train_model, width=20).pack(pady=5)
Button(root, text="Predict Failures (Test Data)", command=predict_test_data, width=25).pack(pady=5)
Button(root, text="Reset", command=reset, width=20).pack(pady=5)

# Dynamic Input Section
Label(root, text="Enter Dynamic Input Data:", font=("Arial", 12)).pack(pady=5)
dynamic_inputs_frame = Frame(root)
dynamic_inputs_frame.pack(pady=5)
dynamic_labels = ["Temperature", "Vibration", "Pressure", "Operating Hours"]
dynamic_inputs = [Entry(dynamic_inputs_frame, width=10) for _ in dynamic_labels]

for i, entry in enumerate(dynamic_inputs):
    Label(dynamic_inputs_frame, text=dynamic_labels[i]).pack(side='left', padx=5)
    entry.pack(side='left', padx=5)

Button(root, text="Predict from Dynamic Input", command=predict_dynamic_input, width=25).pack(pady=10)





# Run the GUI
root.mainloop()
