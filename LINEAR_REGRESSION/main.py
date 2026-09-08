import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

w = 1
b = 1
predicted_output_array = np.array([], dtype=float)
error_array = np.array([], dtype=float)

predicted_output_array = w * x + b

error_array = y - predicted_output_array

total_error = np.sum(error_array ** 2)

gradient_w = -2 * np.sum(x * error_array) / len(x)
gradient_b = -2 * np.sum(error_array) / len(x)

print("Predictions:", predicted_output_array)
print("Errors:", error_array)
print("Squared Errors:", total_error)
print("MSE:", total_error/len(error_array))
print("Gradient w:", gradient_w)
print("Gradient b:", gradient_b)

learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):

    predicted_output = w * x + b

    error = y - predicted_output

    mse = np.mean(error ** 2)

    gradient_w = -2 * np.sum(x * error) / len(x)
    gradient_b = -2 * np.sum(error) / len(x)

    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

    print(f"Epoch {epoch + 1}: w={w:.4f}, b={b:.4f}, MSE={mse:.4f}")
