# Deep-learning
This Python code implements logic gates (NOT, AND, OR, XOR, NAND) using a perceptron model. It includes a unit step function for binary outputs and a perceptron model to compute results based on inputs, weights, and bias. Each gate is modular, with test cases to validate functionality.
import numpy as np

-------------------# Define the unit step function-----------------
def unit_step(value):                 
    """Returns 1 if value is non-negative, else returns 0."""
    return 1 if value >= 0 else 0

# Perceptron model implementation
def perceptron_model(inputs, weights, bias):
    """Calculates the output of the perceptron model."""
    total_input = np.dot(weights, inputs) + bias
    return unit_step(total_input)

# Logic functions defined using the perceptron model

def NOT(x):
    """Implements the NOT logic function."""
    return perceptron_model(x, weights=np.array([-1]), bias=0.5)

def AND(x):
    """Implements the AND logic function."""
    return perceptron_model(x, weights=np.array([1, 1]), bias=-1.5)

def OR(x):
    """Implements the OR logic function."""
    return perceptron_model(x, weights=np.array([1, 1]), bias=-0.5)

def NAND(x):
    """Implements the NAND logic function."""
    return perceptron_model(x, weights=np.array([-1, -1]), bias=1.5)

def XOR(x):
    """Implements the XOR logic function using AND, OR, and NOT."""
    y1 = AND(x)  # AND output
    y2 = OR(x)   # OR output
    y3 = NOT(y1)  # NOT of the AND output
    final_input = np.array([y2, y3])  # Combine OR output and NOT of AND
    return AND(final_input)  # Final AND for XOR

# Testing the logic functions with different inputs
test_cases = [
    (np.array([0, 1]), "XOR"),
    (np.array([1, 1]), "XOR"),
    (np.array([0, 0]), "XOR"),
    (np.array([1, 0]), "XOR"),
    (np.array([0, 0]), "NAND"),
    (np.array([1, 0]), "NAND"),
    (np.array([0, 1]), "NAND"),
    (np.array([1, 1]), "NAND")
]

for inputs, gate in test_cases:
    if gate == "XOR":
        result = XOR(inputs)
    elif gate == "NAND":
        result = NAND(inputs)
    print(f"{gate}({inputs[0]},{inputs[1]}) = {result}")


