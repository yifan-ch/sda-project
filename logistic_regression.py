import numpy as np
import matplotlib.pyplot as plt
import data_model
from data_model import df_mean, df_z_scores
from pathlib import Path
from env import PATHS
from sklearn.model_selection import train_test_split


def initialize_parameters(num_features):
    """
    Initialize weights and bias for logistic regression.
    """
    # Initialize weights to zeros
    weights = np.zeros((num_features, 1))
    # Initialize bias to zero
    bias = 0.0
    return weights, bias


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagation(X, weights, bias):
    """
    Calculate the predicted probabilities using logistic regression.
    """
    # Linear combination: z = X.w + b
    z = np.dot(X, weights) + bias
    # z = np.clip(z, -10, 10)
    # print(f"Linear combination {z}:")
    # Apply the sigmoid function
    predictions = sigmoid(z)
    # print(f"predictions: {predictions}")
    return predictions


def compute_loss(y_true, y_pred):
    m = y_true.shape[0]

    # Avoid division by zero by clipping predicted probabilities
    # y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    epsilon = 1e-9
    # Debugging the predictions and log terms
    # print("Clipped predictions:", y_pred)
    # print("Log terms:", y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Binary cross-entropy loss
    loss = -(1 / m) * np.sum(
        y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)
    )
    return loss

    # y1 = y_true * np.log(y_pred + epsilon)
    # y2 = (1-y_true) * np.log(1 - y_pred + epsilon)
    # return -np.mean(y1 + y2)


def backpropagation(X, y, y_pred):
    """
    Perform backpropagation to compute gradients of the loss.
    """
    num_samples = X.shape[0]
    # Derivative of sigmoid and bce X.T
    dz = y_pred - y
    # Compute the gradient of the loss with respect to weights (w)
    dw = (1 / num_samples) * np.dot(X.T, dz)
    # Compute the gradient of the loss with respect to bias (b)
    db = (1 / num_samples) * np.sum(dz)
    return dw, db


def update_parameters(weights, bias, dw, db, learning_rate):
    """
    Update weights and bias using gradient descent.
    """
    weights -= learning_rate * dw
    bias -= learning_rate * db
    return weights, bias


def predict_classes(y_pred, threshold=0.4):
    """
    Convert predicted probabilities to class labels.
    """
    return (y_pred >= threshold).astype(int)


def calculate_metrics(y_true, y_pred_labels):
    """
    Calculate accuracy, false positives, false negatives, true positives, and true negatives.
    """
    # True Positives
    TP = np.sum((y_true == 1) & (y_pred_labels == 1))
    # False Positives
    FP = np.sum((y_true == 0) & (y_pred_labels == 1))
    # False Negatives
    FN = np.sum((y_true == 1) & (y_pred_labels == 0))
    # True Negatives
    TN = np.sum((y_true == 0) & (y_pred_labels == 0))

    # Accuracy calculation
    accuracy = (TP + TN) / len(y_true)

    return accuracy, TP, FP, FN, TN


def train_logistic_regression(X, y, num_epochs, learning_rate):
    """
    Train logistic regression using gradient descent.
    """
    num_features = X.shape[1]
    weights, bias = initialize_parameters(num_features)
    losses = []

    for epoch in range(num_epochs):
        # Forward propagation
        y_pred = forward_propagation(X, weights, bias)

        # Compute loss
        losses.append(compute_loss(y, y_pred))

        # Backpropagation
        dw, db = backpropagation(X, y, y_pred)

        # Update parameters
        weights, bias = update_parameters(weights, bias, dw, db, learning_rate)

    return weights, bias, losses


def train_and_evaluate(X, y, num_epochs, learning_rate, random_state):
    """
    Train and evaluate logistic regression on a dataset with a given random_state for data splitting.
    """
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    # print(f"\nTraining and evaluating with random_state={random_state}")

    # Train the model on the training set
    weights, bias, losses = train_logistic_regression(X_train, y_train, num_epochs, learning_rate)

    # Evaluate the model on the test set
    y_test_pred = forward_propagation(X_test, weights, bias)
    y_test_pred_labels = predict_classes(y_test_pred)

    # Calculate metrics
    accuracy, TP, FP, FN, TN = calculate_metrics(y_test, y_test_pred_labels)

    # Print the metrics for this run
    # print(f"Test Accuracy: {accuracy * 100:.2f}%")
    # print(f"True Positives: {TP}")
    # print(f"True Negatives: {TN}")
    # print(f"False Positives: {FP}")
    # print(f"False Negatives: {FN}")

    return accuracy, TP, FP, FN, TN, losses

def run_logistic_regression():
    z_scores = df_z_scores()
    X = z_scores.drop(columns=["status"]).to_numpy()
    y = z_scores["status"].to_numpy().reshape(-1, 1)  # Reshape to (m, 1) for matrix multiplication

    # Hyperparameters
    num_epochs = 1000
    learning_rate = 0.01

    # List to store results
    accuracies = []
    true_positives = []
    false_positives = []
    false_negatives = []
    true_negatives = []
    all_losses = []

    # Run the training and evaluation 1000 times with different random_state values
    for random_state in range(10):
        accuracy, TP, FP, FN, TN, losses = train_and_evaluate(
            X, y, num_epochs, learning_rate, random_state
        )

        # Store the results for this run
        accuracies.append(accuracy)
        true_positives.append(TP)
        false_positives.append(FP)
        false_negatives.append(FN)
        true_negatives.append(TN)
        all_losses.append(losses[-1])

    # Average metrics across all 1000 runs
    avg_accuracy = np.mean(accuracies)
    avg_true_positive = np.mean(true_positives)
    avg_false_positive = np.mean(false_positives)
    avg_false_negative = np.mean(false_negatives)
    avg_true_negative = np.mean(true_negatives)
    avg_loss = np.mean(all_losses)

    print(f"minimum accuracy: {min(accuracies)}")
    print(f"maximum accuracy: {max(accuracies)}")

    print("Averaged Metrics After 1000 Runs:")
    print(f"Average Accuracy: {(round(avg_accuracy, 2))*100}%")
    print(f"Average True Positives: {avg_true_positive}")
    print(f"Average False Positives: {avg_false_positive}")
    print(f"Average False Negatives: {avg_false_negative}")
    print(f"Average True Negatives: {avg_true_negative}")
    print(f"Average Final Loss: {avg_loss}")

    # Plot the final loss curve for all 1000 runs
    plt.plot(range(10), all_losses)
    plt.xlabel("Run Index")
    plt.ylabel("Final Loss")
    plt.title("Final Loss Across 1000 Runs")
    plt.show()


if __name__ == "__main__":
    run_logistic_regression()
