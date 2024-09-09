# Hate-speech-detention-model

## Overview
This project focuses on building a model to automatically classify tweets into one of three categories:
1. **Hate Speech**: Language that promotes explicit hatred towards individuals or groups.
2. **Offensive Language**: Language that is disrespectful or rude but does not necessarily promote hatred.
3. **Neither Hate nor Offensive**: Neutral language that does not fall into the first two categories.

The model aims to address the growing need for detecting harmful content on social media platforms, particularly hate speech, to maintain healthy and safe online interactions.

## Steps in the Model

### 1. Data Preprocessing
Tweets often contain noisy and irrelevant data such as URLs, special symbols, and unnecessary characters. To ensure optimal performance of the machine learning model, the raw text is cleaned and standardized using the following steps:
- **Lowercasing**: Convert all text to lowercase for uniformity (e.g., "Hate" and "hate" are treated the same).
- **Removing Unwanted Characters**: Remove square brackets, URLs, punctuation, and numbers.
- **Stopword Removal**: Common words like "the," "is," "and" are removed, focusing the model on more meaningful words.
- **Stemming**: Reduce words to their base or root form (e.g., "running" becomes "run").

### 2. Text Vectorization
Machine learning models require numerical input. Using **CountVectorizer**, the text is converted into a matrix of token counts (Bag of Words model), which:
- Tokenizes the text into words.
- Counts word frequency across the dataset.

The result is a sparse matrix, with rows representing tweets and columns representing unique words.

### 3. Data Splitting
The processed data is split into:
- **Training Set (70%)**: Used to train the model.
- **Test Set (30%)**: Used to evaluate the model’s performance on unseen data.

This is done using the **train_test_split** function to ensure generalization and prevent overfitting.

### 4. Classification with Decision Tree Classifier
We use a **Decision Tree Classifier** as the model for classification. It:
- **Splits** the data based on the most informative features (words).
- **Creates a tree-like structure**, where internal nodes represent words and leaf nodes represent the class (hate speech, offensive, or neutral).

Decision trees are simple and interpretable but may require fine-tuning to avoid overfitting.

### 5. Model Training and Evaluation
The model is trained on the training set:
- **Training**: The model learns the patterns from tweets and associates word frequencies with the respective labels.
- **Testing**: The model is evaluated on the test set, and accuracy is calculated using the `score` function.

### 6. Prediction Function for New Tweets
A function is provided for predicting the class of new tweets:
- **User Input**: The user can input a new tweet.
- **Preprocessing**: The input is preprocessed in the same way as the training data (cleaning, vectorization).
- **Prediction**: The decision tree classifier predicts whether the tweet is hate speech, offensive, or neutral.

## Technologies Used
- Python
- Scikit-learn
- Pandas
- Numpy
- CountVectorizer
- DecisionTreeClassifier
