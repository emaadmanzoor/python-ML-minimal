"""
    Data:       logistic_regression_data.txt
    Algorithm:  Simple logistic regression
"""

import sys
import math


def scalefeatures(data, m, n):
    mean = [0] + [(sum([data[i][j] for i in xrange(m)])) / float(m)
                for j in xrange(1, n + 1)]

    stddeviation = [0] + [math.sqrt(sum([(data[i][j] - mean[j])
                                        ** 2 for i in xrange(m)])
                                    / float(m))
                          for j in xrange(1, n + 1)]

    for j in xrange(1, n + 1):
        for i in xrange(m):
            data[i][j] = (data[i][j] - mean[j]) / stddeviation[j]

    return data


def h_logistic_regression(theta, x, n):
    theta_t_x = sum([theta[i] * x[i] for i in xrange(n + 1)])
    try:
        k = 1.0 / (1 + math.exp(-theta_t_x))
    except OverflowError:
        if theta_t_x > 10 ** 5:
            k = 1.0 / (1 + math.exp(-100))
        else:
            k = 1.0 / (1 + math.exp(100))
    if k == 1.0:
        k = 0.99999
    return k


def cost_logistic_regression(theta, x, y, m, n):
    summation = sum([y[i] * math.log(h_logistic_regression(theta, x[i], n))
                     + (1 - y[i])
                     * math.log(1 - h_logistic_regression(theta, x[i], n))
                        for i in xrange(m)])
    return -summation / m


def gradientdescent_logistic(theta, x, y, m, n, alpha, iterations):
    for i in xrange(iterations):
        thetatemp = theta[:]
        for j in xrange(n + 1):
            summation = sum([(h_logistic_regression(theta, x[k], n) - y[k])
                             * x[k][j]
                             for k in xrange(m)])
            thetatemp[j] = thetatemp[j] - alpha * summation / m
        theta = thetatemp[:]
    return theta


def main():
    x = []  # List of training example parameters
    y = []  # List of training example results

    for line in sys.stdin:
        data = map(float, line.split(','))
        x.append(data[:-1])
        y.append(data[-1])

    m = len(x)      # Number of training examples
    n = len(x[0])   # Number of features

    # Append a column of 1's to x
    x = [[1] + i for i in x]

    # Initialize theta's
    initialtheta = [0.0] * (n + 1)
    learningrate = 0.001
    iterations = 4000

    x = scalefeatures(x, m, n)

    # Run gradient descent to get our guessed hypothesis
    finaltheta = gradientdescent_logistic(initialtheta,
                                          x, y, m, n,
                                          learningrate, iterations)

    # Evaluate our hypothesis accuracy
    print "Initial cost:", cost_logistic_regression(initialtheta, x, y, m, n)
    print "Final cost:", cost_logistic_regression(finaltheta, x, y, m, n)

if __name__ == "__main__":
    main()
