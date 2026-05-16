"""
LIME explainability analysis for CMKT framework.
"""

import numpy as np

import torch

import matplotlib.pyplot as plt

from lime import lime_image

from skimage.segmentation import mark_boundaries


# ------------------------------------------------------
# Run LIME explanation
# ------------------------------------------------------
def run_lime(
    encoder_net,
    task_net,
    image,
    device,
    num_samples=1000
):

    encoder_net.eval()

    task_net.eval()

    # --------------------------------------------------
    # Prediction function for LIME
    # --------------------------------------------------
    def predict_fn(images):

        # ----------------------------------------------
        # Convert RGB to grayscale if needed
        # ----------------------------------------------

        if images.shape[-1] == 3:

            images_gray = np.mean(
                images,
                axis=-1
            )

        else:

            images_gray = images

        # ----------------------------------------------
        # Convert to tensor
        # ----------------------------------------------

        tensor = torch.tensor(
            images_gray,
            dtype=torch.float32
        )

        tensor = tensor.unsqueeze(1)

        tensor = tensor.to(device)

        # ----------------------------------------------
        # Forward pass
        # ----------------------------------------------

        with torch.no_grad():

            features = encoder_net(
                tensor
            )

            outputs = task_net(
                features
            )

        probs = outputs.cpu().numpy()

        # ----------------------------------------------
        # Create two-class probability output
        # ----------------------------------------------

        probs = np.concatenate(
            [
                1 - probs,
                probs
            ],
            axis=1
        )

        return probs

    # --------------------------------------------------
    # Convert image for LIME
    # --------------------------------------------------

    image_np = image.cpu().numpy()

    image_np = np.squeeze(image_np)

    # ----------------------------------------------
    # LIME expects RGB-like image
    # ----------------------------------------------

    image_rgb = np.stack(
        [
            image_np,
            image_np,
            image_np
        ],
        axis=-1
    )

    # --------------------------------------------------
    # Create explainer
    # --------------------------------------------------

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image_rgb,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples
    )

    # --------------------------------------------------
    # Extract explanation mask
    # --------------------------------------------------

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    # --------------------------------------------------
    # Plot explanation
    # --------------------------------------------------

    plt.figure(figsize=(8, 8))

    plt.imshow(
        mark_boundaries(temp / 255.0, mask)
    )

    plt.title(
        "LIME Explanation"
    )

    plt.axis("off")

    plt.show()

    return explanation