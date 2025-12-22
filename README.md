Counter-UAS Threat Detection Using CNNs

Overview

This project aims to detect potential threat areas from top-down satellite imagery using a custom Convolutional Neural Network (CNN). The model is trained to identify objects such as buildings, fields, and other areas that may pose a high-risk for unauthorized drone activity.

The workflow involves training the CNN on publicly available datasets, including SpaceNet AOI Vegas and the DeepGlobe Land Cover Classification Dataset, followed by fine-tuning for improved accuracy.

This project is implemented in Python and fully version-controlled on GitHub


Features

Custom CNN Architecture: Designed from scratch to detect key features in satellite imagery.

Multi-Dataset Training: Leverages SpaceNet for initial training and DeepGlobe for fine-tuning.

Threat Level Classification: Outputs a risk score based on detected features in the image.

Demo Ready: Supports pre-recorded top-down imagery to simulate real-world testing.

Future Work

Incorporate real-time drone footage integration.

Expand dataset coverage for higher geographic generalization.

Improve threat classification granularity (e.g., low, medium, high).