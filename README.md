# AI-Driven Building Energy Management: A Hybrid CNN-Transformer Approach with K-Means Anomaly Detection

## Acknowledgements

I would like to express my deepest and most sincere gratitude to Dr. Duong Thi Kim Chi for her invaluable guidance, professional mentorship, and continuous support throughout the development of this research project. Her insights into advanced machine learning architectures and power system dynamics were instrumental in refining the methodology and ensuring the academic rigor of this work.

## Abstract

This project presents an end-to-end AI pipeline for hourly power consumption forecasting and anomaly detection in commercial and residential buildings. By integrating a Hybrid CNN-Transformer Encoder for sequence modeling and K-Means Clustering for non-parametric residual analysis, I address the limitations of traditional RNNs and static thresholding methods. The model captures long-range temporal dependencies and local features simultaneously, achieving high F1-scores in detecting power spikes and meter malfunctions.

---

## 1. Introduction & Motivation

Reliable energy forecasting is the backbone of the Smart Grid. Traditional methods like ARIMA or LSTM often struggle with the Global Data context or the vanishing gradient problem when dealing with long sequences. In this project, I implement a Transformer-based approach, which utilizes the Self-Attention mechanism to weigh the importance of different historical time steps (e.g., comparing 8:00 AM today with 8:00 AM yesterday) regardless of their distance in the sequence.

### Why Transformer?

The core advantage lies in the attention formula:

This allows the model to process the entire 48-hour lookback window in parallel, capturing periodicities that sequential models often miss.

---

## 2. Data Pipeline & Systematic Preprocessing

I utilize the Building Data Genome Project 2 (BDG2) dataset, which provides real-world electricity meter data.

### 2.1 Analysis of the Dataset

The data exhibits high volatility and strong daily periodicity. I observed that buildings often have missing values due to sensor failures.

1. Linear Interpolation: To maintain the structural integrity of the time series without introducing artificial noise.
2. Building Selection: I filtered out meters with an average consumption < 10 to ensure the model trains on significant energy patterns rather than background noise.
3. Normalization: I applied Min-Max Scaling to map values into the [0, 1] range, which is critical for the stability of the dot-product attention mechanism.

---

## 3. Methodology: The Hybrid CNN-Transformer Architecture

### 3.1 Architecture Overview

My implementation diverges from the standard Transformer described in the reference paper by incorporating a Convolutional Neural Network (CNN) layer as a learnable feature extractor and positional encoder.

### 3.2 The CNN Component (Conv1D)

Instead of using fixed Sinusoidal Positional Encoding, I employ a Conv1D layer with a Relu activation.

1. Function: It extracts local temporal features (e.g., sudden ramps or drops in the last 3 hours).
2. Justification: CNNs are translation-invariant. By using Conv1D before the Attention block, I provide the model with a Learnable Position mapping, making the spatial-temporal relationship more flexible than fixed mathematical functions.

### 3.3 Transformer Encoder Blocks

I utilized two layers of Encoder blocks. Each block consists of:

1. Multi-Head Attention (MHA): 4 heads to attend to different representation subspaces.
2. Residual Connections (Add): To prevent signal degradation.
3. Layer Normalization: To stabilize the internal dynamics of the network.

---

## 4. Anomaly Detection via K-Means Residual Clustering

A key innovation in this pipeline is the transition from Forecasting to Detection.

### 4.1 Residual Logic

After the model predicts the consumption for T+1, I calculate the Absolute Residual:

### 4.2 K-Means Clustering (k=2)

In the reference paper, thresholds are often manually set. I automated this by applying K-Means to the residuals.

1. Cluster 1 (Normal): Low residuals where the model predicted accurately.
2. Cluster 2 (Anomaly): High residuals where the model failed to predict, indicating a deviation from the learned pattern (a spike, a leak, or a failure).
3. Dynamic Threshold: The threshold is set as the center of the anomaly cluster, allowing the system to adapt to different buildings' noise levels.

### 4.3 Fallback Mechanism

To ensure robustness, I implemented a 4-Sigma Rule (Threshold = mean + 4 * std) as a fallback in case the K-Means clustering does not converge due to extremely clean data.

---

## 5. Comparative Analysis: Code vs. Reference Paper

While the reference paper (Power Consumption Predicting and Anomaly Detection Based on Transformer and K-Means) provides the foundation, my implementation introduces several distinct improvements:

| Feature | Reference Paper | My Implementation (CNN-Transformer) |
| --- | --- | --- |
| Input Data | Multivariate (Voltage, Current, etc.) | Univariate Sequence (Multi-building context) |
| Positioning | Standard Positional Encoding | CNN-based Learnable Position Encoding |
| Dimensionality | Fixed Vector Embedding | GlobalAveragePooling1D for feature compression |
| Anomaly Logic | Threshold based on Score | Clustering-based Dynamic Thresholding |

Technical Reasoning: I chose a CNN-Transformer hybrid because, in building energy, local shapes of power usage (peaks) are just as important as long-term cycles. The Conv1D layer captures these shapes before the Transformer analyzes the daily cycles.

---

## 6. Evaluation & Results

To validate the model, I performed Artificial Fault Injection by doubling values at random 5% intervals.

Metrics achieved:

1. Precision: 0.9615 (Minimal false positives).
2. Recall: 0.8824 (High sensitivity to actual faults).
3. F1-Score: 0.9202 (Excellent balance between precision and recall).

---

## 7. Conclusion

This project demonstrates that a Hybrid CNN-Transformer model, coupled with unsupervised K-Means clustering, provides a highly adaptive solution for energy monitoring. By shifting from static rules to learnable features and dynamic clustering, the system becomes significantly more resilient to the varied consumption patterns of modern infrastructure.

Developed as an academic research project under the supervision of TS. Dương Thị Kim Chi.