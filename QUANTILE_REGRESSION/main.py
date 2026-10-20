import numpy as np


def pinball_loss(residual, tau):
    if residual >= 0:
        return residual * tau
    else:
        return (tau - 1) * residual


def mean_pinball_loss(y_true, y_pred, tau):
    losses = []

    for actual, predicted in zip(y_true, y_pred):
        residual = actual - predicted
        losses.append(pinball_loss(residual, tau))

    return sum(losses) / len(losses)


def pinball_gradient_prediction(residual, tau):
    if residual > 0:
        return -tau
    elif residual < 0:
        return 1 - tau
    else:
        return 0


x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

w = 1.0
b = 1.0
tau = 0.1
learning_rate = 0.01
epochs = 1000

y_pred = w * x + b
initial_loss = mean_pinball_loss(y, y_pred, tau)

print("Initial loss:", initial_loss)

for epoch in range(epochs):

    y_pred = w * x + b
    residuals = y - y_pred

    grad_w = 0.0
    grad_b = 0.0

    for xi, residual in zip(x, residuals):
        grad_pred = pinball_gradient_prediction(
            residual, tau
        )

        grad_w += xi * grad_pred
        grad_b += grad_pred

    grad_w /= len(x)
    grad_b /= len(x)

    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

y_pred = w * x + b
final_loss = mean_pinball_loss(y, y_pred, tau)

print("Final weight:", w)
print("Final bias:", b)
print("Final loss:", final_loss)
print("Predictions:", y_pred)
