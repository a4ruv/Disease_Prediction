# ⚕️ AI-Based Symptom-to-Disease Prediction System

**A supervised machine learning system that uses the Random Forest Classifier to predict potential diseases based on user-selected symptoms. Deployed as a standalone Graphical User Interface (GUI) application using Tkinter and Joblib for real-time, interactive predictions.** 

---

## ✨ Project Overview

This repository contains the full implementation of an AI-based system designed to predict potential diseases from a list of user-provided symptoms. The project was developed as a requirement for the **Artificial Intelligence** course (CSN304).

### Key Objectives

* **Data Preprocessing and Analysis:** Load, clean, and analyze the raw training and testing datasets.
* *Model Development:** Train a robust classification model using the **Random Forest** algorithm.
* **Model Evaluation:** Assess performance using metrics like Accuracy, Precision, Recall, and F1-Score.
* **Deployment:** Create a user-friendly Graphical User Interface (GUI) application to allow users to input symptoms and receive a predicted disease and its probability.

---

## ⚙️ Methodology

### Random Forest Classifier
The **Random Forest** algorithm is the core of this system. It is an ensemble learning method that constructs multiple Decision Trees during training. For classification, the output is determined by the class selected by most trees (a "voting" process). This technique effectively mitigates **overfitting** inherent in single Decision Trees, leading to more generalized and accurate predictions.

### Data
* **Datasets:** Two CSV files, `Training.csv` and `Testing.csv`, are used.
* **Features ($\mathbf{X}$):** 132 symptoms (e.g., 'itching', 'skin\_rash', 'joint\_pain'). The presence of a symptom is encoded as **1** and absence as **0** (One-Hot Encoding format).
* **Label ($\mathbf{y}$):** The target variable is 'prognosis', containing **41 unique diseases** to be predicted.

### Workflow Summary
The application is designed to be persistent:

1.  **Start GUI:** The application attempts to load a pretrained model.
2.  **Model Loading/Training:** If the model files (`disease_model.joblib` and `symptoms_list.joblib`) are not found, the model is trained using `Training.csv` and then saved using **Joblib**.
3.  **Input:** The GUI displays dropdown menus, allowing the user to select **up to 5 symptoms**
4.  **Prediction:** When the "Predict Disease" button is clicked, an input vector of 0s and 1s is created based on the selected symptoms.
5.  **Output:** The Random Forest Classifier uses `predict_proba()` to generate probability scores[cite: 78, 107]. [cite_start]The application then displays the **top 3 predicted diseases with their percentage probabilities**.


[Image of machine learning workflow diagram]

<img width="2048" height="2048" alt="image" src="https://github.com/user-attachments/assets/f16b7e7a-3fab-49b1-8170-af50339337ff" />

---

## 📊 Evaluation Results

The Random Forest model demonstrated very high performance on the testing dataset:

* **Accuracy:** **0.97619** (97.6%) 
* **Macro Average F1-Score:** **0.98** 

The classification report showed near-perfect scores (1.00) for the majority of the 41 diseases, indicating the model is highly effective for this task.

---

## 🚀 Implementation and Deployment

The project is implemented using two main files:

### 1. `Random_Forest_Disease_Prediction.ipynb` (Model Pipeline)
* Handles data loading, cleaning (removes unnamed columns), model instantiation (`RandomForestClassifier(n_estimators=100, random_state=42)`) . training, prediction, and detailed evaluation (Accuracy and Classification Report).

### 2. `disease_gui.py` (GUI Application)
* **Persistence:** Uses `joblib` to save and load the trained model and the list of expected symptoms, preventing retraining on every start.
* **GUI Setup:** Built using the `tkinter` library.
* **Symptom Input:** Creates five `tk.OptionMenu` dropdowns populated with the list of 132 possible symptoms
* **Prediction Function:** The `predict()` function calculates the input vector, calls `model.predict_proba()`, identifies the top 3 diseases, and displays the results in a message box.

---

## 👩‍💻 Project Team

**Submitted by:**
* **Aaruv Choudhary** (1000019997) 
* **Yash Saini** (1000019632) 
* **Vansh Goel** (1000019838) 

**Submitted to:**
* **Dr. [cite_start]Himani Sharma**, Assistant Professor 

**University:** DIT University, Dehradun [cite: 6, 17]
]**Course:** Bachelor of Technology, Computer Science and Engineering 
**Submission Date:** November 2025 
