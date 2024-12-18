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
    """
    Sigmoid function.
    """

    return 1 / (1 + np.exp(-z))


def forward_propagation(X, weights, bias):
    """
    Calculate the predicted probabilities using logistic regression.
    """
    # Linear combination: z = X.w + b
    z = np.dot(X, weights) + bias
    # Apply the sigmoid function
    predictions = sigmoid(z)

    return predictions


def compute_loss(y_true, y_pred):
    """
    Compute the binary cross-entropy loss.
    """

    m = y_true.shape[0]

    # Avoid division by zero by clipping predicted probabilities
    epsilon = 1e-9

    # Binary cross-entropy loss
    loss = -(1 / m) * np.sum(
        y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)
    )

    return loss


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
    df, num_epochs, learning_rate, random_state, fraction_training=0.5, threshold=0.5
):
    """
    Train and evaluate logistic regression on a
    dataset with a given random_state for data splitting.
    """

    X_training, y_training, X_test, y_test = split_training_test(
        df, random_state, fraction_training
    )

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
    df,
    threshold=0.5,
    repetitions=100,
    num_epochs=1000,
    fraction_training=0.5,
    learning_rate=0.01,
    use_elasticnet=False,
):
    """
    Run logistic regression on the dataset and return the average metrics.
    """
    df

    if use_elasticnet:
        df_elasticnet = elastic_net_model(df)
        df_elasticnet["status"] = df["status"]
        df = df_elasticnet

    metrics = []

    # Run the training and evaluation multiple with different random_state values
    for random_state in range(repetitions):
        metric, losses = train_and_evaluate(
            df,
            num_epochs,
            learning_rate,
            random_state,
            fraction_training,
            threshold,
        )

        metrics.append(metric)

    # Average metrics across all runs
    avg_metrics = np.mean(np.array(metrics), axis=0)

    return avg_metrics, losses
