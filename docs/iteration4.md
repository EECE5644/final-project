# Methods
We chose three ML methods for our project: `Multinomial Naive Bayes`, `Linear SVM`, and `Logistic Regression`.
The reason we chose these three methods is that they are all well-suited for text classification tasks.

# Pipelines
Firstly, we need to preprocess the text from 20newsgroups dataset, which includes lowercasing, removing punctuation, removing numbers, removing extra spaces, tokenization, and stopwoard removal.
Secondly, we use vectorizers to convert the text into numerical features. In this step, the data also is normalized and filtered low frequency words.
Thirdly, we use grid search to determine the best hyperparameters for each model.
Finally, we evaluate the performance of each model using metrics such as accuracy, precision, recall, and F1-score.

# Preprocessing
Text preprocessing techniques: lowercasing, removing punctuation, removing numbers, removing extra spaces, tokenization, and stopwoard removal.
The code is in [preprocessing.py](./data_preprossor.py).

# Hyperparameter tuning

We tuned the hyperparameters of each model using grid search with cross-validation.
For `Multinomial Naive Bayes`, we tuned the `alpha` parameter, which controls the smoothing of the model.
For `Linear SVM`, we tuned the `C` parameter, which is inversely related to the regularization strength of the model.
For `Logistic Regression`, we also tuned the `C` parameter, which controls the regularization strength of the model.


| Model | Best Params |
|---|---|
| Multinomial Naive Bayes | alpha=0.01 |
| Linear SVM | C=1 |
| Logistic Regression | C=10 |


# Performance Evaluation

Model comparison on 20 Newsgroups (TF-IDF features, 5-fold CV grid search).

| Model | Best Params | CV Accuracy | Test Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Multinomial Naive Bayes | alpha=0.01 | 0.8653 | 77.34% | 0.7713 | 0.7641 | 0.7649 |
| Linear SVM | C=1 | 0.8543 | 77.00% | 0.7652 | 0.7614 | 0.7618 |
| Logistic Regression | C=10 | 0.8478 | 76.65% | 0.7624 | 0.7583 | 0.7590 |


*Naive Bayes* performs best across all test metrics.

# Visualizations

![Confusion Matrix for Multinomial Naive Bayes](attchments/confusion_matrix_multinomial_naive_bayes.png)
![Confusion Matrix for Linear SVM](attchments/confusion_matrix_linear_svm.png)
![Confusion Matrix for Logistic Regression](attchments/confusion_matrix_logistic_regression.png)
