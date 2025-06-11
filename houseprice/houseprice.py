import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tkinter import *
from tkinter import messagebox, ttk

# Load dataset
try:
    data = pd.read_csv("house_price.csv")  # Make sure this file contains 'area', 'rooms', 'price'
except FileNotFoundError:
    messagebox.showerror("File Error", "Dataset 'house_data.csv' not found.")
    exit()

# Prepare the data
X = data[['area', 'rooms']]
y = data['price']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = r2_score(y_test, model.predict(X_test)) * 100

# -------- GUI Code --------
def predict_price():
    try:
        area = float(area_var.get())
        rooms = int(rooms_var.get())

        input_df = pd.DataFrame({'area': [area], 'rooms': [rooms]})
        prediction = model.predict(input_df)
        result_label.config(
            text=f"Predicted Price:\n₹ {prediction[0]:,.2f}",
            fg="green"
        )
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for Area and Rooms.")

# Setup GUI window
root = Tk()
root.title("🏡 Advanced House Price Predictor")
root.geometry("400x400")
root.resizable(False, False)
root.configure(bg="#e6f2ff")

# Title Label
title_label = Label(
    root,
    text="House Price Prediction",
    font=("Helvetica", 18, "bold"),
    bg="#e6f2ff",
    fg="#333"
)
title_label.pack(pady=20)

# Frame for Inputs
input_frame = Frame(root, bg="#e6f2ff")
input_frame.pack(pady=10)

# Area input
Label(input_frame, text="Area (sqft):", font=("Arial", 12), bg="#e6f2ff").grid(row=0, column=0, padx=10, pady=10, sticky=E)
area_var = StringVar()
Entry(input_frame, textvariable=area_var, font=("Arial", 12), width=15).grid(row=0, column=1, padx=10)

# Rooms input
Label(input_frame, text="Rooms:", font=("Arial", 12), bg="#e6f2ff").grid(row=1, column=0, padx=10, pady=10, sticky=E)
rooms_var = StringVar()
Entry(input_frame, textvariable=rooms_var, font=("Arial", 12), width=15).grid(row=1, column=1, padx=10)

# Predict Button
predict_btn = Button(
    root,
    text="Predict Price",
    font=("Arial", 12, "bold"),
    bg="#4CAF50",
    fg="white",
    padx=10,
    pady=5,
    command=predict_price
)
predict_btn.pack(pady=15)

# Result Label
result_label = Label(
    root,
    text="",
    font=("Arial", 14),
    bg="#e6f2ff"
)
result_label.pack(pady=10)

# Show Accuracy
accuracy_label = Label(
    root,
    text=f"Model Accuracy: {accuracy:.2f}%",
    font=("Arial", 10),
    bg="#e6f2ff",
    fg="#555"
)
accuracy_label.pack(pady=5)

# Run the GUI
root.mainloop()
