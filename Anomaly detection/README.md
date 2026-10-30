# Anomaly Detection in SKAB Data

The `Anomaly_detection.ipynb` notebook analyzes multivariate measurements from a pump system and compares approaches to anomaly detection. The data includes timestamps, vibration, current, pressure, temperature, voltage, and flow measurements, together with anomaly and changepoint labels.

## Analysis workflow

1. Load and explore 46,623 measurements from three days. The notebook examines distributions, correlations, time series, seasonality, and stationarity.
2. Clean the data: rows without labels and duplicate rows are removed. When measurements at the same timestamp have conflicting labels, the ambiguous label is marked missing. Missing timestamps are not imputed; a short interval with missing anomaly labels is removed along with a neighboring seasonal window.
3. Scale features with MinMax and apply PCA. Four components explain more than 92% of the variance.
4. Split the data chronologically into training, test, and validation sets, with 15 minutes reserved for each of test and validation. Models are compared using the F1 score.
5. Compare a dummy classifier, Random Forest, SVM, LSTM, Isolation Forest, and an autoencoder. Hyperparameters are tuned for the supervised models and the two unsupervised models.

## Key observations and results

- The series show a downward trend and are non-stationary. Some features have seasonality, but its pattern is inconsistent across days.
- Most continuous features are correlated. The `changepoint` label is highly imbalanced; among labeled records, `anomaly` is approximately 65% class 0 and 35% class 1.
- The dummy classifier scores F1 = 0.00 on the test set.
- Default Random Forest achieves F1 ≈ 0.47. Tuning improves cross-validation performance but lowers the test score to ≈ 0.41.
- Default SVM achieves F1 = 0.00. After tuning, its test score rises to ≈ 0.42.
- Default LSTM achieves test F1 ≈ 0.61. After tuning, it reaches test F1 ≈ 0.70 and validation F1 ≈ 0.89, the best results among the models compared.
- Tuned Isolation Forest achieves test F1 ≈ 0.25 and validation F1 ≈ 0.46. The tuned autoencoder achieves test F1 ≈ 0.33 and validation F1 ≈ 0.34.
- The final tuned test F1 scores are: Random Forest 0.408, SVM 0.421, LSTM 0.703, Isolation Forest 0.248, and autoencoder 0.334. On validation they are 0.167, 0.389, 0.892, 0.461, and 0.335, respectively.

## Conclusions

The tuned LSTM performs best on both the test and validation splits, suggesting it makes better use of temporal ordering and dependencies in this dataset. The unsupervised Isolation Forest and autoencoder score below the LSTM and tuned SVM on the test split. Tuning does not guarantee better performance on held-out data: Random Forest and SVM test results are weaker than their best cross-validation results. The dataset is short, has gaps, and contains ambiguous labels, so evaluation may be unstable. Treat these figures as a comparison under the notebook's particular split, rather than a general measure of detection quality.

