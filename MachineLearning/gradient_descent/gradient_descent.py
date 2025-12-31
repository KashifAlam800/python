import numpy as np
import math
import pandas as pd

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

x = df["math"].to_numpy()
y = df["cs"].to_numpy()


gradient_descent(x,y)

