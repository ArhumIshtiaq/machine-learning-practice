import numpy as np


def nonlinear(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


def predict(x_test, y_test, ss):
    prediction = np.around(nonlinear(np.dot(x_test, ss)))
    error = np.mean(np.abs(y_test - prediction))
    print("\nP:", prediction, "\n", "\nE:", error)


x = np.array([[150, 1], [185, 0], [180, 1], [140, 0], [200, 1],
              [199, 0], [121, 1], [220, 0], [140, 1], [175, 0],
              [210, 1], [99, 0], [120, 1], [120, 0], [179, 1]])

y = np.array([[0], [0], [1], [0], [1], [0], [0],
              [1], [0], [0], [1], [0], [0], [0], [0]])

np.random.seed(1)

syn0 = 2 * (np.random.random((2, 1)))

for iteration in range(100):
    for _ in range(100000):

        l0 = x
        l1 = nonlinear(np.dot(l0, syn0))

        l1_error = y - l1

        l1_delta = l1_error * nonlinear(l1, deriv=True)

        syn0 += l0.T.dot(l1_delta)

    if (iteration % 10 == 0):
        print("Iteration no.:", iteration)
        print("Error:", np.mean(np.abs(l1_error)))

x_test = np.array([[180, 0], [250, 0], [180, 1], [110, 0], [195, 1], [111, 0]])

y_test = np.array([0, 0, 1, 0, 1, 0]).T

predict(x_test, y_test, syn0)
print("\nSynapse:", syn0)


"""
for _ in range(10000):
prediction = nonlinear(np.dot(x_test,syn0))
error = np.mean(np.abs(y_test - prediction))
error_delta = error * nonlinear(prediction, deriv = True)
syn0 += x_test.T.dot(error_delta)

if (_ == 9999):
print("P:",prediction,"\nE:", error)
print(syn0)
"""
