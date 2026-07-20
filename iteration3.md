# Final Project: Topic Classification on the 20 Newsgroups Dataset
## Project Information
**Team Members:**
  - Dayeon Hyun
  - Yu Zhang
**Project Title:** Topic Classification on the 20 Newsgroups Dataset
**Task type:** Supervised multi-class text classification
**GitHub Repo**: [https://github.com/EECE5644/final-project](https://github.com/EECE5644/final-project)

## Dataset Summary
- **Name:** 20 Newsgroups
- **Source**: scikit-learn
- **Link**: [https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)
- **Size:** ~18,000 posts, spread across 20 categories.
- **Number of Samples:** 18846
- **Target Variable**: 20 classes newsgroups

### Why this dataset?
- It is a classic dataset for text classification, widely used in research and tutorials.
- It has a manageable size for our project, allowing us to experiment with different machine learning models.
- It has a clear, well-studied benchmark, so we can compare our results against known baselines.
- If time permits, it can be extended to a deep-learning approach with RNNs or LSTMs, which can capture sequential information in text.

## Data Preprocessing
- Codes for the preprocessing process for the final project dataset (20 Newsgroups) are in 20newsgroups.ipynb.
- Preprocessing steps we completed for 20 newsgroups dataset:
  1. Encoding Categorical Variables
    - Convert the 20 newgroup category names to binary (0~19)
  2. Handling Missing Values
    - Remove the empty documents(News)
  3. Removing Duplicate Records
    - Remove the duplicate text entires
  4. Detecting and Treating Outliers
    - Remove the documents that were too short (e.g. fewer than 10 words)
  5. Feature engineering
    - Remove the quote and unnecessary info (e.g. headers at first including link kind of stuff)
    - Convert each word into a weighted numerical feature
  6. Feature normalization
    - Normalize each document vector and make each document length (words length) used for the data is consistent
  7. Feature selection
    - Using chi-square and only select top 5000 most discrimivative words of each texts to use as features
  8. Training and testing data split
    - Split the train and test data to 8:2.

## Machine Learning Model
- The algorithm we selected is Logistic Regression (for multi-class classification).
- We also can extend to deep-learning approach, such as RNNs or LSTMs to increase the accuracy.
- We selected the Logistic Regression because it is a strong/common baseline for supervised text classification task. It can handle multi-class classification. It also can efficiently handle large dataset.
- Expected advantages are time efficiency on working with large datasets, reduction on the risk of overfitting (compared to complex model) by using regularization if needed, and solid accuracy.
- Planned hyperparameter tuning:
  - solver: lbfgs
  - max_iter: 1000,
  - random_state: 8888
  - multiclass: multinominal
  - penalty: L2

## Research Questions
- How do feature selection and data preprocessing affect model performance?
- Which features appear to be the strongest predictors? Are there redundant or highly correlated features?
- How effective is Logistic Regression in classifying documents across the 20 Newsgroups dataset categories? Does any other model have higher accuracy?
- Does applying deep learning approch (RNN / LSTM) give meaningful impact on accuracy for classifying the newsgroup for each news?
- If there's news that contains the features of multiple newsgroups and can be considered to be in multiple newsgroups, how model treat this type of news document?
- Are the 20 newsgroups proper to classify the type of news, too much, or too small?
