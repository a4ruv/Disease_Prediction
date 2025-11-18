# disease_gui.py
import os
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL_FILE = "disease_model.joblib"
SYMPTOMS_FILE = "symptoms_list.joblib"
TRAIN_FILE = "Training.csv"

def train_and_save_model():
    print("Training model from", TRAIN_FILE)
    df = pd.read_csv(TRAIN_FILE)
    if "prognosis" not in df.columns:
        raise ValueError("Training.csv must contain a 'prognosis' column.")
    X = df.drop("prognosis", axis=1)
    y = df["prognosis"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(list(X.columns), SYMPTOMS_FILE)
    print("Model trained and saved to", MODEL_FILE)
    return model, list(X.columns)

def load_model_and_symptoms():
    if os.path.exists(MODEL_FILE) and os.path.exists(SYMPTOMS_FILE):
        model = joblib.load(MODEL_FILE)
        symptoms = joblib.load(SYMPTOMS_FILE)
        print("Loaded model and symptoms list from files.")
        return model, symptoms
    else:
        if not os.path.exists(TRAIN_FILE):
            raise FileNotFoundError(f"{TRAIN_FILE} not found. Place it in the same folder.")
        return train_and_save_model()

# Load or train
model, symptoms_list = load_model_and_symptoms()

# GUI
root = tk.Tk()
root.title("AI Disease Prediction")
root.geometry("620x520")
root.resizable(False, False)

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(fill="both", expand=True)

tk.Label(frame, text="AI-Based Disease Prediction", font=("Arial", 18, "bold")).pack(pady=(0,10))
tk.Label(frame, text="Select up to 5 symptoms:", font=("Arial", 12)).pack(anchor="w")

# Create dropdowns
selected_vars = []
for i in range(5):
    var = tk.StringVar(value="Select Symptom")
    opt = tk.OptionMenu(frame, var, *(["Select Symptom"] + symptoms_list))
    opt.config(width=40)
    opt.pack(pady=6)
    selected_vars.append(var)

def predict():
    chosen = [v.get() for v in selected_vars if v.get() != "Select Symptom"]
    if not chosen:
        messagebox.showerror("Input error", "Please select at least one symptom.")
        return

    # create input vector
    input_vec = np.zeros(len(symptoms_list), dtype=int)
    for s in chosen:
        try:
            idx = symptoms_list.index(s)
            input_vec[idx] = 1
        except ValueError:
            pass

    # predict probabilities and get top result (and optionally top 3)
    try:
        preds = model.predict([input_vec])
        top = preds[0]
        # get probabilities for classes (if model supports predict_proba)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([input_vec])[0]
            classes = model.classes_
            top_indices = np.argsort(probs)[::-1][:3]
            top_list = [(classes[i], round(probs[i]*100, 2)) for i in top_indices]
            msg = "Top predictions:\n" + "\n".join([f"{d} — {p}%" for d,p in top_list])
        else:
            msg = f"Predicted disease: {top}"
        messagebox.showinfo("Prediction", msg)
    except Exception as e:
        messagebox.showerror("Prediction error", str(e))

btn = tk.Button(frame, text="Predict Disease", command=predict, font=("Arial", 12, "bold"), width=20)
btn.pack(pady=18)

# Optional: show selected symptoms text area
result_frame = tk.Frame(frame)
result_frame.pack(fill="x", pady=(10,0))
tk.Label(result_frame, text="Selected symptoms:", font=("Arial",10)).pack(anchor="w")
selected_display = tk.Label(result_frame, text="", anchor="w", justify="left")
selected_display.pack(fill="x")

def update_display(*args):
    chosen = [v.get() for v in selected_vars if v.get() != "Select Symptom"]
    selected_display.config(text=", ".join(chosen) if chosen else "None")

for v in selected_vars:
    v.trace_add("write", update_display)

root.mainloop()
