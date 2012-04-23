"""
    Problem:    Predict house selling price
    Data:       house_data.txt
    Algorithm:  Multivariate linear regression
"""

import sys
import math


def scalefeatures(data, m, n):
    mean = [0] + [(sum([data[i][j] for i in xrange(m)])) / float(m)
                  for j in xrange(1, n + 1)]

    stddeviation = [0] + [math.sqrt(sum(
                                        [(data[i][j] - mean[j]) ** 2
                                         for i in xrange(m)]
                                        ) / float(m))
                         for j in xrange(1, n + 1)]

    for j in xrange(1, n + 1):
        for i in xrange(m):
            data[i][j] = (data[i][j] - mean[j]) / stddeviation[j]

    return data


def h(theta, x, n):
    """
        theta: hypothesis [theta0, theta1, ..., theta_n)
        x: training example [1, x1, x2, ..., x_n]
    """
    return sum([theta[i] * x[i] for i in xrange(n + 1)])


def cost(theta, x, y, m, n):
    summation = sum([(h(theta, x[i], n) - y[i]) ** 2
                        for i in xrange(m)])
    return summation / (2 * m)


def gradientdescent(theta, x, y, m, n, alpha, iterations):
    for i in xrange(iterations):
        thetatemp = theta[:]
        for j in xrange(n + 1):
            summation = sum([(h(theta, x[k], n) - y[k]) * x[k][j]
                             for k in xrange(m)])
            thetatemp[j] = thetatemp[j] - alpha * summation / m
        theta = thetatemp[:]
        print theta
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

    # Initialize learning parameters
    initialtheta = [0.0] * (n + 1)
    learningrate = 0.01
    iterations = 100

    x = scalefeatures(x, m, n)

    # Run gradient descent to get our guessed hypothesis
    finaltheta = gradientdescent(initialtheta,
                                 x, y, m, n,
                                 learningrate, iterations)

    # Evaluate our hypothesis accuracy
    print "Initial cost:", cost(initialtheta, x, y, m, n)
    print "Final cost:", cost(finaltheta, x, y, m, n)

if __name__ == "__main__":
    main()
