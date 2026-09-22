# Linear Regression From Scratch

## About this project
I built this personal project to implement linear regression from scratch in Python to apply the statistics knowledge I learned in Cambridge AS Level and have a basic understanding of machine learning.

## What this project includes

* **Manual Math:** Mean Squared Error (MSE) and partial derivatives for slope ($m$) and intercept ($c$).
* **Optimization:** Standard Gradient Descent and an implementation of Adagrad to test adaptive learning rates.
* **Data Processing:** Standardisation using Z-score normalisation, plus converting trained parameters back to the original scale.
* **Visualization:** Graphs for the dataset, fitted line, and loss/error over iterations.
* **Benchmarking:** Direct comparison of predictions against `scikit-learn`’s `LinearRegression`.

## Files

* `main.py` — Main script containing the model, math, testing code, and plots.

## Current Status & Notes

* There is commented-out testing code and dataset generation logic left in `main.py` from experimenting with different ideas.
* Placeholders exist for Adam, which I plan to finish when I revisit optimization algorithms later.
* There are plans to make it support multidimensional datasets and feature selections such as **Ridge** and **Lasso**.
* It’s meant to show my raw process of building a model from the ground up, making mistakes, and figuring out how the underlying math actually works.

## Setup & Running

```bash
# Install dependencies
pip install numpy pandas matplotlib scikit-learn

# Run the script
python main.py
```

*Note: Data generation code is included directly in `main.py` and can be uncommented or modified to generate new test datasets.*

## What I Learned

* How gradient descent actually reduces error step-by-step.
* Why standardising data matters for convergence.
* How optimisers like Adagrad adjust learning rates.
* How to verify a manual implementation against a standard library like `scikit-learn`.
