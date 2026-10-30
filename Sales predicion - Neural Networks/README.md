# Sales Forecasting with Neural Networks

The `Sales_prediction_-_Neural_Networks.ipynb` notebook forecasts daily sales for individual shop–item pairs one month ahead. The data contains sales history, prices, and shop and item identifiers.

## Analysis workflow

- Exploratory analysis examines series length and completeness, sales, prices, weekdays, holidays, outliers, correlations, and seasonality.
- Short and sparse series are filtered out. The analysis retains 51 shops and 153 items; many shop–item pairs do not have enough history.
- Returns are adjusted against earlier sales. Potential outliers are identified using a modified z-score, but recurring peaks for some items are retained as possible promotions or genuine behavior.
- Initial missing months are dropped, and remaining gaps are imputed with methods suited to each variable. Calendar and holiday features, a 35-day lag, and 35-day rolling mean and standard deviation are created.
- The forecast horizon is 31 days. Seasonal Naive is compared with NBEATSx, LSTM, and TFT. Neural model hyperparameters are optimized, and models are also evaluated with recursive forecasts on the validation set.
- The evaluation metric is MAPE; lower values indicate smaller relative errors.

## Results

- Seasonal Naive achieves an overall MAPE of ≈ 0.988 (about 99%).
- With default settings, LSTM and TFT achieve approximately 62% and 63% MAPE, respectively; NBEATSx exceeds 84%.
- After optimization on the test set: NBEATSx ≈ 68.7%, LSTM ≈ 54.6%, TFT ≈ 57.5%. LSTM has the best test result.
- On recursive validation forecasts, the ranking changes: NBEATSx ≈ 55.4%, LSTM ≈ 70.3%, and TFT ≈ 61.6%. NBEATSx performs best here, while LSTM does worse than on the test set.

## Conclusions

Optimization improves results over the simple seasonal baseline, but errors remain high. The best model depends on the evaluation period: LSTM leads on the test set, while NBEATSx leads on recursive validation. Many series have low daily sales (0–10 units), so an error of just one or two units can substantially increase MAPE. Interpret results with this property of the data and metric in mind.


