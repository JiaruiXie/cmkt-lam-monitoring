"""
Trainer for cross-modal semantic alignment framework.
"""

import torch
import torch.nn as nn

from src.models.semantic_alignment import (
    pairwise_y,
    pairwise_X
)


class Trainer:

    def __init__(
        self,
        encoder_net,
        task_net,
        encoder_optimizer,
        classifier_optimizer,
        device,
        margin=1.0,
        gamma=0.5
    ):

        self.encoder_net = encoder_net

        self.task_net = task_net

        self.encoder_optimizer = (
            encoder_optimizer
        )

        self.classifier_optimizer = (
            classifier_optimizer
        )

        self.device = device

        self.margin = margin

        self.gamma = gamma

        self.EPS = 1e-8

    # --------------------------------------------------
    # Classification loss
    # --------------------------------------------------
    def classification_loss(
        self,
        predictions,
        labels
    ):

        labels = labels.view(-1, 1)

        # ------------------------------------------
        # Weighted BCE
        # ------------------------------------------

        weight_negative = 3.0

        weight_positive = 1.0

        weights = torch.full(
            labels.shape,
            weight_negative,
            device=self.device
        )

        weights[labels == 1] = (
            weight_positive
        )

        criterion = nn.BCELoss(
            weight=weights
        )

        loss = criterion(
            predictions.view(-1, 1),
            labels
        )

        return loss

    # --------------------------------------------------
    # Contrastive semantic alignment loss
    # --------------------------------------------------
    def semantic_alignment_loss(
        self,
        visual_features,
        audio_features,
        labels
    ):

        # ------------------------------------------
        # Pairwise label distance
        # ------------------------------------------

        dist_y = pairwise_y(
            labels,
            labels
        )

        # ------------------------------------------
        # Pairwise feature distance
        # ------------------------------------------

        dist_X = pairwise_X(
            visual_features,
            audio_features
        )

        # ------------------------------------------
        # Contrastive loss
        # ------------------------------------------

        contrastive_loss = torch.sum(
            dist_y * torch.maximum(
                torch.tensor(
                    0.0,
                    device=self.device
                ),
                self.margin - dist_X
            ),
            1
        ) / (
            torch.sum(
                dist_y,
                1
            ) + self.EPS
        )

        contrastive_loss += torch.sum(
            (1 - dist_y) * dist_X,
            1
        ) / (
            torch.sum(
                1 - dist_y,
                1
            ) + self.EPS
        )

        contrastive_loss = torch.mean(
            contrastive_loss
        )

        contrastive_loss *= 0.5

        return contrastive_loss

    # --------------------------------------------------
    # Train one epoch
    # --------------------------------------------------
    def train_epoch(
        self,
        train_loader
    ):

        self.encoder_net.train()
        
        self.task_net.train()
        
        correct = 0

        total = 0
        
        total_encoder_loss = 0.0

        total_alignment_loss = 0.0

        total_task_loss = 0.0

        for batch in train_loader:

            # ======================================
            # Load batch
            # ======================================

            visual = batch["visual"].to(
                self.device
            )

            audio = batch["audio"].to(
                self.device
            )

            labels = batch["label"].float().to(
                self.device
            )

            # ======================================
            # Zero gradients
            # ======================================

            self.encoder_optimizer.zero_grad()

            self.classifier_optimizer.zero_grad()

            # ======================================
            # Forward pass
            # ======================================

            visual_features = (
                self.encoder_net(visual)
            )

            audio_features = (
                self.encoder_net(audio)
            )

            visual_predictions = (
                self.task_net(
                    visual_features
                )
            )

            audio_predictions = (
                self.task_net(
                    audio_features
                )
            )

            # ======================================
            # Classification loss
            # ======================================

            task_loss = (
                self.classification_loss(
                    visual_predictions,
                    labels
                )
            )

            task_loss += (
                self.classification_loss(
                    audio_predictions,
                    labels
                )
            )

            task_loss /= 2

            # ======================================
            # Semantic alignment loss
            # ======================================

            alignment_loss = (
                self.semantic_alignment_loss(
                    visual_features,
                    audio_features,
                    labels
                )
            )

            # ======================================
            # Total loss
            # ======================================

            encoder_loss = (
                self.gamma * task_loss
                +
                (1 - self.gamma)
                * alignment_loss
            )

            # ======================================
            # Backpropagation
            # ======================================

            encoder_loss.backward()

            self.encoder_optimizer.step()

            self.classifier_optimizer.step()

            # ======================================
            # Statistics
            # ======================================

            total_encoder_loss += (
                encoder_loss.item()
            )

            total_alignment_loss += (
                alignment_loss.item()
            )

            total_task_loss += (
                task_loss.item()
            )

            # --------------------------------------
            # Compute training accuracy
            # --------------------------------------

            visual_binary_predictions = (
                visual_predictions > 0.5
            ).float()

            correct += (
                visual_binary_predictions.view(-1)
                ==
                labels.view(-1)
            ).sum().item()

            total += labels.size(0)

        return {
            "encoder_loss":
                total_encoder_loss
                / len(train_loader),

            "alignment_loss":
                total_alignment_loss
                / len(train_loader),

            "task_loss":
                total_task_loss
                / len(train_loader),

            "accuracy":
                correct / total
        }

    # --------------------------------------------------
    # Full training loop
    # --------------------------------------------------
    def fit(
        self,
        train_loader,
        epochs
    ):

        history = {
            "encoder_loss": [],
            "alignment_loss": [],
            "task_loss": [],
            "accuracy": []
        }

        for epoch in range(epochs):

            print(
                f"Epoch {epoch+1}/{epochs}"
            )

            metrics = self.train_epoch(
                train_loader
            )

            history["encoder_loss"].append(
                metrics["encoder_loss"]
            )

            history["alignment_loss"].append(
                metrics["alignment_loss"]
            )

            history["task_loss"].append(
                metrics["task_loss"]
            )

            history["accuracy"].append(
                metrics["accuracy"]
            )

            print(
                f"Encoder Loss: "
                f"{metrics['encoder_loss']:.4f}"
            )

            print(
                f"Alignment Loss: "
                f"{metrics['alignment_loss']:.4f}"
            )

            print(
                f"Task Loss: "
                f"{metrics['task_loss']:.4f}"
            )

            print(
                f"Accuracy: "
                f"{metrics['accuracy']:.4f}"
            )

        return history