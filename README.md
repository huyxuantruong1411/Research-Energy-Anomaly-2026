# AI-Driven Building Energy Management  
## A Hybrid CNN-Transformer Approach with K-Means Anomaly Detection

---

## Acknowledgements

I would like to express my deepest and most sincere gratitude to **Dr. Duong Thi Kim Chi** for her invaluable guidance, professional mentorship, and continuous support throughout the development of this research project. Her insights into advanced machine learning architectures and power system dynamics were instrumental in refining the methodology and ensuring the academic rigor of this work.

---

## Abstract

This project presents an end-to-end AI pipeline for hourly power consumption forecasting and anomaly detection in commercial and residential buildings. By integrating a **Hybrid CNN-Transformer Encoder** for sequence modeling and **K-Means Clustering** for non-parametric residual analysis, the study addresses the limitations of traditional RNNs and static thresholding methods. The model captures long-range temporal dependencies and local features simultaneously, achieving high F1-scores in detecting power spikes and meter malfunctions.

---

## 1. Introduction & Motivation

Reliable energy forecasting is the backbone of the Smart Grid. Traditional methods such as ARIMA or LSTM often struggle with the *global data context* or the vanishing gradient problem when dealing with long sequences.  

In this project, a **Transformer-based approach** is implemented, leveraging the **Self-Attention mechanism** to weigh the importance of different historical time steps (e.g., comparing 8:00 AM today with 8:00 AM yesterday) regardless of their distance in the sequence.

### Why Transformer?

The core advantage lies in the attention formulation:

\[
Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

This allows the model to process the entire 48-hour lookback window in parallel, capturing periodicities that sequential models often miss.

---

## 2. Data Pipeline & Systematic Preprocessing

The **Building Data Genome Project 2 (BDG2)** dataset is utilized, providing real-world electricity meter data.

### 2.1 Analysis of the Dataset

The data exhibits high volatility and strong daily periodicity. Missing values are frequently observed due to sensor failures.

- **Linear Interpolation**: Maintains structural integrity of the time series without introducing artificial noise.  
- **Building Selection**: Meters with average consumption < 10 are filtered out to avoid training on background noise.  
- **Normalization**: Min-Max Scaling is applied to map values into the \([0, 1]\) range, which is critical for the stability of the dot-product attention mechanism.

---

## 3. Methodology: The Hybrid CNN-Transformer Architecture

### 3.1 Architecture Overview

The implementation diverges from the standard Transformer by incorporating a **Convolutional Neural Network (CNN)** layer as a learnable feature extractor and positional encoder.

### 3.2 The CNN Component (Conv1D)

Instead of fixed sinusoidal positional encoding, a **Conv1D layer with ReLU activation** is employed.

- **Function**: Extracts local temporal features (e.g., sudden ramps or drops within the last 3 hours).  
- **Justification**: CNNs are translation-invariant. By applying Conv1D before the attention block, the model gains a *learnable positional mapping*, offering greater flexibility than fixed mathematical encodings.

### 3.3 Transformer Encoder Blocks

Two encoder layers are utilized, each consisting of:

- **Multi-Head Attention (MHA)**: 4 heads to attend to different representation subspaces.  
- **Residual Connections (Add)**: Prevent signal degradation.  
- **Layer Normalization**: Stabilizes internal network dynamics.

---

## 4. Anomaly Detection via K-Means Residual Clustering

A key innovation is the transition from **forecasting** to **detection**.

### 4.1 Residual Logic

After predicting consumption at \(T+1\), the absolute residual is computed:

\[
R_t = |Y_{actual} - Y_{predicted}|
\]

### 4.2 K-Means Clustering (\(k = 2\))

Rather than manually defining thresholds, K-Means is applied to residuals:

- **Cluster 1 (Normal)**: Low residuals where predictions are accurate.  
- **Cluster 2 (Anomaly)**: High residuals indicating spikes, leaks, or failures.

**Dynamic Threshold**: Defined as the center of the anomaly cluster (\(\mu_{anomaly}\)), allowing adaptation to different building noise levels.

### 4.3 Fallback Mechanism

A **4-Sigma Rule** is implemented as a safeguard:

\[
Threshold = mean + 4 \times std
\]

This is used when K-Means fails to converge due to extremely clean data.

---

## 5. Comparative Analysis: Code vs. Reference Paper

While the reference paper *“Power Consumption Predicting and Anomaly Detection Based on Transformer and K-Means”* provides the foundation, several improvements are introduced:

| Feature | Reference Paper | My Implementation (CNN-Transformer) |
|------|----------------|-------------------------------------|
| Input Data | Multivariate (Voltage, Current, etc.) | Univariate Sequence (Multi-building context) |
| Positioning | Standard Positional Encoding | CNN-based Learnable Position Encoding |
| Dimensionality | Fixed Vector Embedding | GlobalAveragePooling1D |
| Anomaly Logic | Threshold-based | Clustering-based Dynamic Thresholding |

**Technical Reasoning**: In building energy data, local power usage *shapes* (peaks) are as important as long-term cycles. Conv1D captures these shapes before the Transformer models daily and weekly patterns.

---

## 6. Evaluation & Results

To validate robustness, **Artificial Fault Injection** was performed by doubling values at random 5% intervals.

**Metrics Achieved**:

- **Precision**: 0.9615  
- **Recall**: 0.8824  
- **F1-Score**: 0.9202  

These results indicate excellent balance between sensitivity and false alarm control.

---

## 7. Conclusion

This project demonstrates that a **Hybrid CNN-Transformer model**, coupled with **unsupervised K-Means clustering**, provides a highly adaptive solution for building energy monitoring. By replacing static rules with learnable features and dynamic thresholds, the system becomes significantly more resilient to diverse consumption patterns in modern infrastructure.

---

*Developed as an academic research project under the supervision of **TS. Dương Thị Kim Chi***.
