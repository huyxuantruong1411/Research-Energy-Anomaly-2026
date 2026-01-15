Power Consumption Forecasting and Anomaly Detection: A Hybrid CNN-Transformer and Unsupervised Clustering Approach



Acknowledgements



I would like to express my deepest gratitude to Dr. Duong Thi Kim Chi for her invaluable guidance, mentorship, and support throughout the development of this project. Her expertise in the field of Artificial Intelligence and Data Science has been a cornerstone in shaping the academic rigor and technical depth of this research.



Abstract



This project presents an end-to-end AI pipeline for hourly power consumption forecasting and point anomaly detection. By shifting from traditional sequential models (RNNs/LSTMs) to a Hybrid CNN-Transformer architecture, I leverage both local feature extraction and global temporal dependency modeling. The system integrates a Transformer Encoder for high-fidelity forecasting and an unsupervised K-Means clustering algorithm for dynamic residual-based anomaly detection. Experimental results on the Building Data Genome Project 2 (BDG2) dataset demonstrate that this hybrid approach effectively handles the volatility of energy loads while maintaining a high F1-score in detecting artificial and real-world anomalies.



1\. Data Pipeline and Exploratory Analysis



1.1 Dataset Specification



I utilized the Building Data Genome Project 2 (BDG2), a large-scale open-source dataset containing non-residential building energy meter data.



Temporal Resolution: Hourly (1-hour intervals).



Metric: Electricity consumption (kWh).



Periodicity: The data exhibits dual seasonality—circadian (24-hour) and weekly (5-day work week vs. weekend) patterns.



1.2 Preprocessing Methodology



To ensure model stability and prevent gradient saturation in the Attention mechanism, I implemented the following:



Missing Value Imputation: Linear interpolation was chosen to maintain the slope of energy trends during sensor downtime.



Noise Filtering: Meters with an average load of $< 10$ units were pruned to eliminate inactive or faulty sensors that act as outliers.



Min-Max Scaling: Normalized all inputs to the $\[0, 1]$ range.



Sliding Window: I constructed a 48-hour lookback window to predict the $t+1$ hour. This window size is critical as it captures two full cycles of daily consumption, allowing the model to learn the "history of the trend" rather than just the immediate previous value.



2\. Architectural Deep Dive: CNN-Transformer vs. Vanilla Transformer



2.1 The Hybrid CNN-Transformer Design



A significant departure from the reference paper and standard Transformer implementations is my integration of 1D Convolutional Neural Networks (CNN) as a learnable embedding layer.



The Architecture:



Local Feature Extraction (Conv1D): Instead of standard linear projections or fixed Sine/Cosine positional encodings, I use a Conv1D layer.



Why? Time series data often contains local patterns (e.g., sudden spikes or drops over 2-3 hours). CNNs are mathematically superior at extracting these local translation-invariant features before they are passed to the global attention mechanism.



Transformer Encoder Blocks: I implemented a 2-block encoder with Multi-Head Self-Attention (MHA).



The Attention Mechanism:





$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d\_k}}\\right)V$$





This allows the model to "attend" to specific hours in the past (e.g., the consumption at 2 PM yesterday) while predicting the consumption for 2 PM today.



Global Average Pooling: Unlike the reference paper that might use the last hidden state, I use Global Average Pooling to aggregate the temporal information across the entire 48-hour window, providing a "holistic" summary of energy usage behavior.



3\. Anomaly Detection Logic



3.1 Unsupervised Residual Clustering



The core innovation in the detection phase is the move away from static thresholds. In energy systems, a "spike" of 10kWh might be normal for a factory but an anomaly for a small office.



Residual Calculation: I compute the Absolute Residual $R = |Y\_{actual} - Y\_{predicted}|$.



K-Means Partitioning ($k=2$): I cluster the residuals into two groups:



Cluster 0 (Normal): Small residuals where the model predicted accurately.



Cluster 1 (Anomaly): Large residuals where the actual data deviated significantly from the learned "normal" trend.



Dynamic Thresholding: The threshold is defined as the center of the Anomaly cluster ($Threshold = \\text{max}(centers)$). This allows the system to be building-specific.



3.2 Statistical Fallback (The 4-Sigma Rule)



To handle edge cases where K-Means might converge on a too-sensitive threshold (due to low variance in residuals), I implemented a fallback:





$$\\text{If } Threshold < (\\mu + 2\\sigma) \\implies Threshold = \\mu + 4\\sigma$$





This ensures that the system does not trigger false positives in highly stable environments.



4\. Comparison with the Reference Paper



My implementation refines several aspects of the methodology presented in the reference paper ("Power Consumption Predicting and Anomaly Detection Based on Transformer and K-Means"):



Feature



Reference Paper Approach



My Implementation (CNN-Transformer)



Embedding



Standard Linear/Positional Encoding



1D-CNN Learnable Embedding



Why?



Traditional Sinusoidal encoding is non-adaptive.



Conv1D learns to filter noise and extract local temporal features specific to the BDG2 dataset.



Pooling



Often Flatten or Last State



Global Average Pooling



Why?



Flattening can lead to overfitting on specific window indices.



GAP creates a translation-invariant representation of the 48-hour sequence.



Residual Logic



K-Means on Error



K-Means + 4-Sigma Fallback



Why?



K-Means can fail if the data is "too normal."



The 4-Sigma rule acts as a safety guardrail for high-precision detection.



5\. Experimental Results



5.1 Forecasting Performance



The model was trained for 15 epochs with an Early Stopping callback. The validation loss stabilized quickly, indicating that the Transformer effectively learned the 24-hour periodicity.



MAE: Significantly lower than the LSTM baseline.



RMSE: Reflects high sensitivity to peak consumption hours.



5.2 Anomaly Metrics (Artificial Stress Test)



I injected artificial anomalies (scaling values by $2.0$) to test the K-Means classifier:



Precision: $0.9615$ (Low false alarm rate).



Recall: $0.8824$ (Detected nearly all injected spikes).



F1-Score: $0.9202$ (High overall harmonic mean of accuracy).



6\. Conclusion



By integrating the local spatial awareness of CNNs with the global temporal awareness of Transformers, I have developed a robust framework for energy monitoring. The unsupervised nature of the K-Means post-processing makes this solution highly scalable across different building types without the need for manual threshold tuning.



Technical Stack



Framework: TensorFlow / Keras



Hardware: NVIDIA T4 GPU (Google Colab)



Libraries: Pandas, Scikit-Learn, Matplotlib, Seaborn

