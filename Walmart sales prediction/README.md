# Walmart Sales Forecasting

This project analyzes Walmart's daily product sales and forecasts the next 28 days using the M5 dataset. The notebook combines sales history with calendar information, event indicators, weekly prices, SNAP availability, and product and store identifiers.

## Notebook workflow

- Load and join the M5 sales, calendar, and price data for 10 stores across California, Texas, and Wisconsin.
- Explore sales and prices by state, category, and department, including distributions, correlations, trends, seasonality, and stationarity.
- Cluster product-store series using recent sales behavior. The analysis finds two groups: a large group of lower-volume series and a smaller group of higher-volume series (silhouette score: 0.85).
- Prepare forecasting features, including day of week, month, year, event counts, SNAP availability, price, a missing-price indicator, a 28-day sales lag, and later 28-day rolling statistics.
- Compare statistical baselines (Seasonal Naive, Holt-Winters, and Poisson regression), Prophet, LSTM, and LightGBM using WRMSSE. Lower WRMSSE indicates better forecasts.
- Tune models on a subset of series, refine LightGBM features, and generate a validation forecast.

## Results

| Model | WRMSSE | Evaluation split |
|---|---:|---|
| Seasonal Naive | 1.3402 | Test, full data |
| Holt-Winters | 1.1547 | Test, full data |
| Poisson regression | 1.3023 | Test, full data |
| Holt-Winters | 1.1259 | Validation, full data |
| Prophet, univariate | 1.1709 | Test, 15% subset |
| Prophet, multivariate | 1.1583 | Test, 15% subset |
| Holt-Winters | 1.1528 | Test, 15% subset |
| LSTM | 1.6403 | Test, 15% subset |
| LightGBM, initial features | 1.1572 | Test, 15% subset |
| LightGBM, rolling features and tuning | 1.0876 | Test, 15% subset |
| LightGBM, retrained with test data | 1.0082 | Validation, 15% subset |

The subset-based results should be interpreted separately from the full-data baseline results. The notebook reports that rolling mean and standard deviation features improve LightGBM on the test subset. Its feature importance analysis also highlights recent sales history and price. Holt-Winters is the strongest of the reported full-data baseline models. Prophet performs similarly to Holt-Winters on the subset, while the LSTM produces relatively flat forecasts and has the highest WRMSSE among the reported subset models.

Residual checks for sampled LightGBM series indicate autocorrelation, non-constant variance, and departures from normality. These findings suggest that the model does not capture all structure in the sales series.

## Conclusions and next steps

The experiments show that feature engineering can improve forecasting quality: adding rolling sales statistics lowers LightGBM's test WRMSSE from 1.1572 to 1.0876 on the sampled series. The validation result is 1.0082 for the retrained model. These results are promising, though comparisons across the full dataset and the 15% subset are not directly equivalent.

Further improvement could come from adding other sales lags and longer rolling windows, selecting features more systematically, and expanding hyperparameter searches. More time and computational power would make it possible to run broader experiments, tune more complex models, and assess improvements more thoroughly. Therefore, the reported results are a useful starting point and can be further improved with additional time and compute.


