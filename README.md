# Political Ideology Prediction from Congressional Tweets

This project focuses on predicting the ideological positions of U.S. Congressional politicians based on their Twitter activity. Using Natural Language Processing (NLP) techniques and machine learning models, the project aims to predict two key dimensions of political ideology: the general liberal-conservative spectrum (Dimension 1) and a second dimension that captures more specific ideological differences (Dimension 2), such as regional issues.

## Table of Contents
- [Introduction](#introduction)
- [Data Description](#data-description)
- [Methods](#methods)
- [Modeling Approach](#modeling-approach)
- [Optimization](#optimization)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The goal of this project is to accurately predict the political alignment of U.S. Congress members based on their tweet data. The model uses a combination of text preprocessing techniques, feature engineering, and machine learning algorithms to predict two ideological dimensions from the tweets.

We achieved an RMSE of **0.2639**, significantly outperforming the baseline RMSE of **0.36**. By leveraging **BERT embeddings** and **XGBoost**, we were able to create a model that captures the nuances of political bias in the tweets.

## Data Description
The dataset consists of **469,740 tweets** from U.S. Congress members, spanning the years 2008–2020. The key features used in the model include:
- **Id**: Unique index number for each tweet.
- **favorite_count**: Number of times the tweet was favorited.
- **retweet_count**: Number of times the tweet was retweeted.
- **full_text**: Full content of the tweet.
- **hashtags**: List of hashtags used in the tweet.
- **year**: Year the tweet was posted.
- **dim1_nominate**: First dimension of the political ideology score (liberal to conservative).
- **dim2_nominate**: Second dimension of the political ideology score (regional and issue-based differences).

## Methods

### A. Text Preprocessing
1. **Byte String Decoding**: Decoded the byte string representation of tweets into human-readable text.
2. **HTML Decoding**: Used the `html` module to decode HTML entities.
3. **Tokenization & Stopword Removal**: Tokenized the text and removed stopwords.
4. **Punctuation Removal**: Replaced punctuation with an empty string.
5. **Short Word Filter**: Removed words shorter than three characters to reduce noise.

### B. Feature Engineering
- **Prediction Bias**: A novel feature generated using **DistilBERT** to predict whether a tweet was written by a Democrat or a Republican.
- **BERT and Hashtag Embeddings**: Semantic embeddings were generated from both the tweet content and hashtags using **BERT** to capture the contextual meaning.

### C. Text Embedding and Vectorization
1. **TF-IDF Vectorization**: Applied TF-IDF to quantify the importance of words in the dataset.
2. **BERT Sentence Embeddings**: Generated semantic embeddings using **BERT** to capture rich contextual information.
3. **Feature Concatenation**: Combined **TF-IDF vectors**, **BERT embeddings**, **hashtags**, and **Prediction Bias** into a single feature set for model training.

## Modeling Approach
- **XGBoost Regressor**: Employed XGBoost for its efficiency in handling large, high-dimensional datasets. Separate models were trained to predict both ideological dimensions.
- **DistilBERT Fine-Tuning**: Fine-tuned a **DistilBERT** model to classify tweets as Democratic or Republican, achieving **90% accuracy**.
- **Combined Features**: The model combined **BERT embeddings** with additional features and used XGBoost to predict the **NOMINATE** dimensions. 

## Optimization
- **GPU Utilization**: Optimized the code to reduce sentence embedding generation time from **240 minutes to 12 minutes** by efficiently using **A100 GPUs** and batch processing techniques.

## Results
- **RMSE**: Achieved an RMSE of **0.2639** for predicting the ideological dimensions, outperforming the benchmark RMSE of **0.36**.
- **Accuracy**: The fine-tuned DistilBERT model achieved **90% accuracy** in predicting the political alignment of tweets.
- **Optimization**: Reduced sentence embedding generation time from **240 minutes to 12 minutes** through GPU optimization and batch processing.
