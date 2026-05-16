"""
Evaluation utilities for CMKT framework.
"""

import torch

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------
# Evaluate model using visual modality only
# ------------------------------------------------------
def evaluate_model(
    encoder_net,
    task_net,
    dataloader,
    device
):

    encoder_net.eval()

    task_net.eval()

    predictions = []

    probabilities = []

    ground_truths = []

    with torch.no_grad():

        for batch in dataloader:

            # ------------------------------------------
            # Load visual modality only
            # ------------------------------------------

            visual = batch["visual"].to(
                device
            )

            labels = batch["label"].to(
                device
            )

            # ------------------------------------------
            # Forward pass
            # ------------------------------------------

            visual_features = encoder_net(
                visual
            )

            outputs = task_net(
                visual_features
            )

            # ------------------------------------------
            # Convert probabilities
            # ------------------------------------------

            probs = outputs.view(-1)

            preds = (
                probs > 0.5
            ).float()

            # ------------------------------------------
            # Store results
            # ------------------------------------------

            probabilities.extend(
                probs.cpu().numpy()
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            ground_truths.extend(
                labels.cpu().numpy()
            )

    # --------------------------------------------------
    # Convert to numpy arrays
    # --------------------------------------------------

    predictions = np.array(
        predictions
    )

    probabilities = np.array(
        probabilities
    )

    ground_truths = np.array(
        ground_truths
    )

    # --------------------------------------------------
    # Compute metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        ground_truths,
        predictions
    )

    precision = precision_score(
        ground_truths,
        predictions
    )

    recall = recall_score(
        ground_truths,
        predictions
    )

    f1 = f1_score(
        ground_truths,
        predictions
    )

    cm = confusion_matrix(
        ground_truths,
        predictions
    )

    report = classification_report(
        ground_truths,
        predictions
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\nEvaluation Results")

    print("-" * 40)

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")

    print(cm)

    print("\nClassification Report")

    print(report)

    # --------------------------------------------------
    # Return metrics
    # --------------------------------------------------

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": predictions,
        "probabilities": probabilities,
        "ground_truths": ground_truths
    }