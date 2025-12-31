import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("test_scores.csv")


def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000000
    n = len(x)
    learning_rate = 0.0002
    cost = 0

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost_pre = cost
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))
        closeness = math.isclose(cost, cost_pre, rel_tol=1e-15)
        if closeness:
            break

    graph_plot(m_curr,b_curr)

def graph_plot(m_curr,b_curr):
    c = df["math"]
    d = df["cs"]

    # Scatter plot
    plt.scatter(c, d, color='blue', label='Actual CS Scores')

    # Example regression line
    y_pred_line = m_curr * c + b_curr

    plt.plot(x, y_pred_line, color='red', label='Predicted Line')
    plt.xlabel("Math Score")
    plt.ylabel("CS Score")
    plt.title("CS Score Prediction from Math Score")
    plt.legend()
    plt.show()


x = df["math"].to_numpy()
y = df["cs"].to_numpy()


gradient_descent(x,y)
