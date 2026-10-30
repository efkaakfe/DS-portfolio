# Movie Review Classification with Classical Machine Learning

The `Text_Classification_ML.ipynb` notebook builds a sentiment classifier using the Stanford NLP IMDb review dataset. The labels are negative (0) and positive (1); the training and test sets each contain 25,000 balanced examples.

## Analysis workflow

1. Explore label balance, review lengths, and word frequencies. The notebook checks for HTML markup, punctuation, letter case, and text length.
2. Preprocess the text by lowercasing, removing HTML markup, punctuation, numbers, and stop words, and applying lemmatization.
3. Convert text to TF-IDF features and classify it with Multinomial Naive Bayes.
4. Split the training data into training and validation subsets, tune parameters with cross-validation, and evaluate the final model on a separate test set.

## Results

- Baseline: 86.5% validation accuracy and 83.0% test accuracy (test macro-F1 ≈ 0.829).
- After optimization: 86.3% validation accuracy and 83.3% test accuracy (test macro-F1 ≈ 0.832).
- The optimized TF-IDF vectorizer uses 19,534 features instead of the default of about 76,000. The test improvement is small but measurable.

## Conclusions

The simple TF-IDF + Naive Bayes pipeline achieves about 83% accuracy on a balanced test set. Performance is similar for both classes, with no clear preference for positive or negative reviews. Text cleaning affects the vocabulary and can change model performance; in particular, removing the word “not” can make phrases such as “not good” harder to interpret.

## Data and code

Main notebook: `Text_Classification_ML.ipynb`. The IMDb dataset is downloaded through the Hugging Face `datasets` library. The notebook also uses the local `training` and `preprocessing` modules.
