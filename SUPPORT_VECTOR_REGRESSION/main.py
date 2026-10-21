def epsilon_insensitive_loss(actual, predicted, epsilon):
    error = abs(actual - predicted)

    if error < epsilon:
        return 0
    else:
        return error - epsilon

def mean_epsilon_loss(y_true, y_pred, epsilon):
    losses = []

    for actual, predicted in zip(y_true, y_pred):
        each_epsilon_insensitive_loss = epsilon_insensitive_loss(
            actual, predicted, epsilon
        )
        losses.append(each_epsilon_insensitive_loss)

    return sum(losses) / len(losses)

y_true = [10, 20, 30, 40]
y_pred = [9.8, 18, 31, 39.5]

for epsilon in [0.0, 0.5, 1.0, 1.5, 2.0]:
    loss = mean_epsilon_loss(y_true, y_pred, epsilon)
    print("Epsilon:", epsilon, "| Mean loss:", loss)
