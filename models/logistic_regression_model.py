"""
Functions for training and testing a logistic regression model.
"""

import numpy as np
from tools.model_tools import split_training_test, predict_class, calculate_metrics
from models.elastic_net_model import elastic_net_model


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


def train_logistic_regression(X, y, num_epochs, learning_rate):
    """
    Train logistic regression using gradient descent.
    """
    num_features = X.shape[1]
    weights, bias = initialize_parameters(num_features)
    losses = []

    for _ in range(num_epochs):
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
    df, num_epochs, learning_rate, random_state, frac_training=0.5, threshold=0.5
):
    """
    Train and evaluate logistic regression on a
    dataset with a given random_state for data splitting.
    """

    X_training, y_training, X_test, y_test = split_training_test(df, random_state, frac_training)

    # Train the model on the training set
    weights, bias, losses = train_logistic_regression(
        X_training, y_training, num_epochs, learning_rate
    )

    # Evaluate the model on the test set
    y_test_pred = forward_propagation(X_test, weights, bias)
    y_test_pred_labels = predict_class(y_test_pred, threshold=threshold)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_test_pred_labels)

    return metrics, losses


def run_logistic_regression(
    df, threshold=0.5, num_reps=100, num_epochs=1000, frac_training=0.5, use_elasticnet=False
):
    z_scores = df

    if use_elasticnet:
        z = elastic_net_model(z_scores)
        z["status"] = z_scores["status"]
        z_scores = z

    # z_scores["status"] = z_scores["status"]
    # X = z_scores2.drop(columns=["status"]).to_numpy()
    # y = z_scores2["status"].to_numpy().reshape(-1, 1)  # Reshape for matrix multiplication

    learning_rate = 0.01
    # List to store results

    # Run the training and evaluation multiple with different random_state values

    metrics = []
    # all_losses = []

    for random_state in range(num_reps):
        metric, losses = train_and_evaluate(
            z_scores,
            num_epochs,
            learning_rate,
            random_state,
            frac_training,
            threshold,
        )

        metrics.append(metric)
        # all_losses.append(losses[-1])

    # Average metrics across all runs
    avg_metrics = np.mean(np.array(metrics), axis=0)

    return avg_metrics, losses
