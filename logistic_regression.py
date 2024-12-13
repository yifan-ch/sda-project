import numpy as np
import matplotlib.pyplot as plt
import data_model
from data_model import df_z_scores
from pathlib import Path
from env import PATHS
from sklearn.model_selection import train_test_split
from multiple_regression_model import split
import pandas as pd
from elastic_net import elastic_net_model


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


def predict_classes(y_pred, threshold=0.25):
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
    # print(f"num epochs: {num_epochs}")
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


def train_and_evaluate(
    X, y, num_epochs, learning_rate, random_state, frac_training=0.5, threshold=0.5
):
    """
    Train and evaluate logistic regression on a dataset with a given random_state for data splitting.
    """
    z_scores = df_z_scores()

    # Split the data for status 0 (64 total samples)
    df_0 = data_model.status(z_scores, 0)
    df_0_training, df_0_test = split(
        df_0, frac_training
    )  # Divide evenly between training and testing

    # Split the data for status 1 (188 total samples)
    df_1 = data_model.status(z_scores, 1)
    df_1_test = df_1.sample(
        n=df_0_test, random_state=random_state
    )  # Select 32 samples for the test set
    df_1_training = df_1.drop(df_1_test.index)  # The rest go into the training set

    # Combine the training data
    df_training = pd.concat([df_0_training, df_1_training], axis=0)
    y_training = df_training["status"].values.reshape(-1, 1)
    x_training = df_training.drop(["status"], axis=1).values

    # Combine the test data
    df_test = pd.concat([df_0_test, df_1_test], axis=0)
    y_test = df_test["status"].values.reshape(-1, 1)
    X_test = df_test.drop(["status"], axis=1).values

    # Split the dataset into training and test sets
    # X_train, X_test2, y_train, y_test2 = train_test_split(
    #     X, y, test_size=0.3, random_state=random_state
    # )

    # print(f"y_training: {y_training.shape}")
    # print(f"x_training: {(x_training.shape)}")
    # print(f"y_test: {(y_test.shape)}")
    # print(f"x_test: {(X_test.shape)}")
    # print(f"X_train: {(X_train.shape)}")
    # print(f"X_test2: {(X_test2.shape)}")
    # print(f"y_train: {(y_train.shape)}")
    # print(f"y_test2: {(y_test2.shape)}")

    # print(f"\nTraining and evaluating with random_state={random_state}")

    # Train the model on the training set
    weights, bias, losses = train_logistic_regression(
        x_training, y_training, num_epochs, learning_rate
    )

    # Evaluate the model on the test set
    y_test_pred = forward_propagation(X_test, weights, bias)
    y_test_pred_labels = predict_classes(y_test_pred, threshold=threshold)

    # Calculate metrics
    accuracy, TP, FP, FN, TN = calculate_metrics(y_test, y_test_pred_labels)

    return accuracy, TP, FP, FN, TN, losses


def run_logistic_regression(threshold=0.5, num_reps=100, num_epochs=1000):
    z_scores = df_z_scores()
    # X = z_scores.drop(columns=["status"]).to_numpy()
    # y = z_scores["status"].to_numpy().reshape(-1, 1) # Reshape for matrix multiplication
    z_scores2 = elastic_net_model()
    print(z_scores.shape[1])
    print(z_scores2.shape[1])
    z_scores2["status"] = z_scores["status"]
    X = z_scores2.drop(columns=["status"]).to_numpy()
    y = z_scores2["status"].to_numpy().reshape(-1, 1) # Reshape for matrix multiplication

    learning_rate = 0.001
    # List to store results
    accuracies = []
    true_positives = []
    false_positives = []
    false_negatives = []
    true_negatives = []
    all_losses = []

    # Run the training and evaluation 1000 times with different random_state values
    for random_state in range(num_reps):
        accuracy, TP, FP, FN, TN, losses = train_and_evaluate(
            X, y, num_epochs, learning_rate, random_state, threshold
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

    print(f"Averaged Metrics After {num_reps} Runs:")
    print(f"Average Accuracy: {(round(avg_accuracy, 2))*100}%")
    print(f"Average True Positives: {avg_true_positive}")
    print(f"Average True Negatives: {avg_true_negative}")
    print(f"Average False Positives: {avg_false_positive}")
    print(f"Average False Negatives: {avg_false_negative}")
    print(f"Average Final Loss: {avg_loss}")

    # Plot the final loss curve for all 1000 runs
    plt.plot(losses)
    plt.xlabel("Run Index")
    plt.ylabel("Final Loss")
    plt.title(f"Final Loss Across {num_reps} Runs")
    plt.savefig("results/logistic-regression/loss_function.png")
    # plt.show()
    return avg_accuracy, losses

def accuracy_per_epoch():
    iterations = iterations = np.arange(100, 4001, 100, dtype=int)
    accuracies = []
    final_losses = []
    
    for num_epochs in iterations:
        accuracy, losses = run_logistic_regression(
            threshold=0.25, num_reps=300, num_epochs=num_epochs
        )
        accuracies.append(accuracy)
        final_losses.append(losses[-1])

    fig, ax1 = plt.subplots()

    # Plot accuracies on the primary y-axis
    ax1.plot(iterations, accuracies, label="Accuracy", color="blue", marker="o")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_title("Accuracy and Loss vs. Epochs")

    # Plot losses on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(iterations, final_losses, label="Loss", color="red", marker="x")
    ax2.set_ylabel("Loss", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Add legends
    plt.savefig("results/logistic-regression/accuracy_per_epoch.png")
    plt.show()


if __name__ == "__main__":
    # run_logistic_regression(threshold=0.25, num_reps=100, num_epochs=1000)
    accuracy_per_epoch()