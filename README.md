# ⚕️ AI-Based Symptom-to-Disease Prediction System

**A supervised machine learning system that uses the Random Forest Classifier to predict potential diseases based on user-selected symptoms. [cite_start]Deployed as a standalone Graphical User Interface (GUI) application using Tkinter and Joblib for real-time, interactive predictions.** [cite: 46, 47, 48]

---

## ✨ Project Overview

[cite_start]This repository contains the full implementation of an AI-based system designed to predict potential diseases from a list of user-provided symptoms[cite: 46]. [cite_start]The project was developed as a requirement for the **Artificial Intelligence** course (CSN304)[cite: 1, 2].

### Key Objectives

* [cite_start]**Data Preprocessing and Analysis:** Load, clean, and analyze the raw training and testing datasets[cite: 50].
* [cite_start]**Model Development:** Train a robust classification model using the **Random Forest** algorithm[cite: 51].
* [cite_start]**Model Evaluation:** Assess performance using metrics like Accuracy, Precision, Recall, and F1-Score[cite: 52].
* [cite_start]**Deployment:** Create a user-friendly Graphical User Interface (GUI) application to allow users to input symptoms and receive a predicted disease and its probability[cite: 53, 54].

---

## ⚙️ Methodology

### Random Forest Classifier
[cite_start]The **Random Forest** algorithm is the core of this system[cite: 47]. [cite_start]It is an ensemble learning method that constructs multiple Decision Trees during training[cite: 60]. [cite_start]For classification, the output is determined by the class selected by most trees (a "voting" process)[cite: 61]. [cite_start]This technique effectively mitigates **overfitting** inherent in single Decision Trees, leading to more generalized and accurate predictions[cite: 62].

### Data
* [cite_start]**Datasets:** Two CSV files, `Training.csv` and `Testing.csv`, are used[cite: 64].
* [cite_start]**Features ($\mathbf{X}$):** 132 symptoms (e.g., 'itching', 'skin\_rash', 'joint\_pain')[cite: 64]. [cite_start]The presence of a symptom is encoded as **1** and absence as **0** (One-Hot Encoding format)[cite: 65].
* [cite_start]**Label ($\mathbf{y}$):** The target variable is 'prognosis', containing **41 unique diseases** to be predicted[cite: 66].

### Workflow Summary
The application is designed to be persistent:

1.  [cite_start]**Start GUI:** The application attempts to load a pretrained model[cite: 69].
2.  [cite_start]**Model Loading/Training:** If the model files (`disease_model.joblib` and `symptoms_list.joblib`) are not found, the model is trained using `Training.csv` and then saved using **Joblib**[cite: 70, 71, 73, 99, 101].
3.  [cite_start]**Input:** The GUI displays dropdown menus, allowing the user to select **up to 5 symptoms**[cite: 74, 105].
4.  [cite_start]**Prediction:** When the "Predict Disease" button is clicked, an input vector of 0s and 1s is created based on the selected symptoms[cite: 76, 106].
5.  [cite_start]**Output:** The Random Forest Classifier uses `predict_proba()` to generate probability scores[cite: 78, 107]. [cite_start]The application then displays the **top 3 predicted diseases with their percentage probabilities**[cite: 79, 108].


[Image of machine learning workflow diagram]


---

## 📊 Evaluation Results

The Random Forest model demonstrated very high performance on the testing dataset:

* [cite_start]**Accuracy:** **0.97619** (97.6%) [cite: 90, 271]
* [cite_start]**Macro Average F1-Score:** **0.98** [cite: 91, 236]

[cite_start]The classification report showed near-perfect scores (1.00) for the majority of the 41 diseases, indicating the model is highly effective for this task[cite: 92, 93].

---

## 🚀 Implementation and Deployment

The project is implemented using two main files:

### 1. `Random_Forest_Disease_Prediction.ipynb` (Model Pipeline)
* [cite_start]Handles data loading, cleaning (removes unnamed columns) [cite: 87, 218][cite_start], model instantiation (`RandomForestClassifier(n_estimators=100, random_state=42)`) [cite: 87, 225][cite_start], training, prediction, and detailed evaluation (Accuracy and Classification Report)[cite: 87].

### 2. `disease_gui.py` (GUI Application)
* [cite_start]**Persistence:** Uses `joblib` to save and load the trained model and the list of expected symptoms, preventing retraining on every start[cite: 98, 99, 101, 102].
* [cite_start]**GUI Setup:** Built using the `tkinter` library[cite: 48, 103].
* [cite_start]**Symptom Input:** Creates five `tk.OptionMenu` dropdowns populated with the list of 132 possible symptoms[cite: 104].
* [cite_start]**Prediction Function:** The `predict()` function calculates the input vector, calls `model.predict_proba()`, identifies the top 3 diseases, and displays the results in a message box[cite: 106, 107, 108, 109].

---

## 👩‍💻 Project Team

**Submitted by:**
* [cite_start]**Aaruv Choudhary** (1000019997) [cite: 13]
* [cite_start]**Yash Saini** (1000019632) [cite: 14]
* [cite_start]**Vansh Goel** (1000019838) [cite: 15]

**Submitted to:**
* **Dr. [cite_start]Himani Sharma**, Assistant Professor [cite: 10, 11]

[cite_start]**University:** DIT University, Dehradun [cite: 6, 17]
[cite_start]**Course:** Bachelor of Technology, Computer Science and Engineering [cite: 4, 5]
[cite_start]**Submission Date:** November 2025 [cite: 20]
