import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
import math
import cmath

class ScientificCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Scientific Calculator")
        self.root.geometry("500x400")
        self.expression = ""

        # Display field
        self.input_text = tk.StringVar()
        input_frame = tk.Frame(root, bd=5, relief=tk.RIDGE)
        input_frame.pack(pady=10, fill='x', padx=10)

        input_field = tk.Entry(input_frame, font=('Arial', 20), textvariable=self.input_text, width=40, bd=3, insertwidth=4)
        input_field.pack(fill='x')

        # Button frame
        btn_frame = tk.Frame(root)
        btn_frame.pack(padx=10)

        buttons = [
            ['7', '8', '9', '/', 'sqrt'],
            ['4', '5', '6', '*', 'log'],
            ['1', '2', '3', '-', 'sin'],
            ['0', '.', '%', '+', 'cos'],
            ['(', ')', 'j', '**', 'tan'],
            ['Clear', 'Del', 'Solve', 'pi', 'exp']
        ]

        for i, row in enumerate(buttons):
            for j, btn in enumerate(row):
                tk.Button(btn_frame, text=btn, width=8, height=2, font=('Arial', 12),
                          command=lambda b=btn: self.on_click(b)).grid(row=i, column=j, padx=2, pady=2)

        # Output area
        self.output_label = tk.Label(root, text="", font=("Arial", 14), fg="blue")
        self.output_label.pack(pady=10)

    def on_click(self, char):
        if char == 'Clear':
            self.expression = ""
            self.output_label.config(text="")
        elif char == 'Del':
            self.expression = self.expression[:-1]
        elif char == 'Solve':
            self.solve_expression()
        else:
            self.expression += str(char)

        self.input_text.set(self.expression)

    def solve_expression(self):
        try:
            result = eval(self.expression, {"__builtins__": None}, {
                "sqrt": np.sqrt, "log": np.log10, "sin": np.sin, "cos": np.cos,
                "tan": np.tan, "pi": math.pi, "exp": np.exp, "j": 1j,
                "complex": complex, "abs": abs, "math": math, "np": np, "cmath": cmath
            })

            if isinstance(result, complex):
                formatted = f"{result.real:.2f} + {result.imag:.2f}j"
            else:
                formatted = str(result)
            self.output_label.config(text=f"Result: {formatted}")
            self.expression = formatted
        except Exception as e:
            messagebox.showerror("Error", f"Invalid Expression:\n{str(e)}")
            self.expression = ""
            self.output_label.config(text="")

# Run the calculator
if __name__ == "__main__":
    root = tk.Tk()
    calc = ScientificCalculator(root)
    root.mainloop()

