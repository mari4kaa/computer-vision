import numpy as np
import matplotlib.pyplot as plt

# Dataset setup
def data_x():
    # Example symbols defined as binary raster images
    symbols = {
        'A': [
            0, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1
        ],
        'B': [
            1, 1, 1, 0, 0,
            1, 0, 0, 1, 0,
            1, 1, 1, 0, 0,
            1, 0, 0, 1, 0,
            1, 1, 1, 0, 0
        ],
        'C': [
            0, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 1,
            0, 1, 1, 1, 0
        ],
        'D': [
            1, 1, 1, 0, 0,
            1, 0, 0, 1, 0,
            1, 0, 0, 0, 1,
            1, 0, 0, 1, 0,
            1, 1, 1, 0, 0
        ],
        'E': [
            1, 1, 1, 1, 0,
            1, 0, 0, 0, 0,
            1, 1, 1, 0, 0,
            1, 0, 0, 0, 0,
            1, 1, 1, 1, 0
        ],
        'F': [
            1, 1, 1, 1, 0,
            1, 0, 0, 0, 0,
            1, 1, 1, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0
        ]
    }

    x = [np.array(data).reshape(1, 25) for data in symbols.values()]

    plt.figure(figsize=(10, 2))
    for i, (key, data) in enumerate(symbols.items()):
        plt.subplot(1, 6, i + 1)
        plt.imshow(1 - np.array(data).reshape(5, 5), cmap='gray')  # Inverted colors
        plt.title(key)
        plt.axis('off')
    plt.show()

    return x

# Labels
def data_y():
    y = np.eye(6)  # One-hot encoding for 6 classes
    return y

# Weight initialization
def generate_weights(input_size, output_size):
    return np.random.randn(input_size, output_size) * 0.1

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Forward pass
def forward(x, w1, w2):
    z1 = x.dot(w1)
    a1 = sigmoid(z1)
    z2 = a1.dot(w2)
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Backward pass
def backprop(x, y, z1, a1, z2, a2, w1, w2, alpha):
    d2 = a2 - y
    d1 = np.multiply(d2.dot(w2.T), sigmoid_derivative(z1))

    w2_update = a1.T.dot(d2)
    w1_update = x.T.dot(d1)

    w2 -= alpha * w2_update
    w1 -= alpha * w1_update

    return w1, w2

# Loss function
def loss(a2, y):
    return np.mean(np.square(a2 - y))

# Training the network
def train(x, y, w1, w2, alpha, epochs, patience):
    accuracy_history = []
    loss_history = []
    best_accuracy = 0
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0
        for i in range(len(x)):
            z1, a1, z2, a2 = forward(x[i], w1, w2)
            epoch_loss += loss(a2, y[i])
            w1, w2 = backprop(x[i], y[i], z1, a1, z2, a2, w1, w2, alpha)
            if np.argmax(a2) == np.argmax(y[i]):
                correct_predictions += 1

        loss_history.append(epoch_loss / len(x))
        accuracy = correct_predictions / len(x) * 100
        accuracy_history.append(accuracy)

        # Early Stopping Logic
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(x):.4f}")
    return w1, w2, loss_history, accuracy_history

# Prediction
def predict(x, w1, w2):
    _, _, _, a2 = forward(x, w1, w2)
    return np.argmax(a2)

# Visualizing results
def evaluate(x, y, w1, w2):
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    print("Expected\tPredicted\tCorrect")
    for i in range(len(x)):
        expected = labels[np.argmax(y[i])]
        predicted = labels[predict(x[i], w1, w2)]
        correct = expected == predicted
        print(f"{expected}\t\t\t{predicted}\t\t\t{correct}")

# Main execution
if __name__ == "__main__":
    x = data_x()
    y = data_y()

    print("Dataset loaded.")

    w1 = generate_weights(25, 10)  # Hidden layer with 10 neurons
    w2 = generate_weights(10, 6)   # Output layer with 6 neurons

    print("Training the network...")
    w1, w2, loss_history, accuracy_history = train(x, y, w1, w2, alpha=0.1, epochs=400, patience=1000)

    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(accuracy_history)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    print("Evaluating the network...")
    evaluate(x, y, w1, w2)
