# 📰 Stock News Sentiment Analysis – Part 2: Advanced Modeling with Embeddings

This repository contains **Part 2** of the *Stock News Sentiment Analysis* project, building on the exploratory and baseline modeling work completed in Part 1.  
In this phase, we focus on **advanced text representation techniques** using word embeddings and transformer-based models to enhance the accuracy of sentiment prediction from financial news.

---

## 📘 Project Overview

Financial markets are influenced not only by quantitative factors like prices and volumes but also by **qualitative sentiment** embedded in financial news.  
This project aims to analyze how news sentiment correlates with stock behavior and identify the most effective **embedding-based predictive models** for market sentiment classification.

In Part 2, we:
- Implement **Word2Vec**, **GloVe**, and **Sentence Transformer** embeddings.
- Train **Gradient Boosting** models on these embeddings.
- Compare model performance and interpret results.
- Tune the models for performance with **Randomized Grid Search**
- Provide recommendations for improving future model performance.

---

## 🧩 Relation to Part 1

| Phase | Focus | Key Outcome |
|-------|--------|--------------|
| **Part 1** | Exploratory Data Analysis (EDA) + Baseline Model (Naive Bayes) | Found clear sentiment trends and established baseline performance |
| **Part 2** | Embedding-based Modeling (Word2Vec, GloVe, SentenceTransformer) | Improved contextual understanding of news and identified best embedding-model combination |

---

## 📂 Repository Structure

```
Stock-News-Sentiment-Analysis/
│
├── 📓 Stock news sentiment analysis - Part 2.ipynb # Advanced modeling notebook (this file)
│
├── 📁 data/
│ ├── stock_news.csv # Financial news & stock price dataset
│
├── 📄 README.md # Documentation file
```

---

## 🧠 Data Description

The dataset combines **financial news headlines** with **daily stock price data** (Open, High, Low, Close, and Volume).  
It captures the sentiment polarity (Positive, Neutral, Negative) of each day’s news, allowing for correlation analysis with stock price movements.

| **Column** | **Description** |
|-------------|----------------|
| `Date` | Date of the market news |
| `News` | Financial news text or headline |
| `Open`, `High`, `Low`, `Close` | Daily stock price indicators |
| `Volume` | Number of shares traded |
| `Label` | Sentiment label: Positive (1), Neutral (0), Negative (-1) |

The data was derived from publicly available sources such as **Kaggle** and **Yahoo Finance**, curated for educational and research purposes.

---

## 🧮 Modeling Approach

### **1. Text Representation**
Three embedding techniques were used to encode financial news:
- **Word2Vec:** Learns word relationships through context-based vectorization.
- **GloVe:** Captures global word co-occurrence patterns for richer representation.
- **Sentence Transformer :** Produces sentence-level embeddings that encode contextual meaning.

### **2. Machine Learning Model**
- A **Gradient Boosting Classifier** was trained on embeddings from each method.
- Models were evaluated on **training** and **validation** datasets.

### **3. Evaluation Metrics**
Model performance was assessed using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score** (primary metric to handle class imbalance)

---

## 📊 Key Findings

- All models exhibited **some level of overfitting**, as seen from discrepancies between training and validation performance.
- The **GloVe-encoded Gradient Boosting model** achieved the **highest F1 Score**, outperforming Word2Vec and SentenceTransformer embeddings.
- **Negative sentiment** was consistently linked with lower stock prices, reaffirming the relationship between investor mood and market movement.
- The **Sentence Transformer** model captured deeper semantic nuances but required more data for stable generalization.

---

## 💡 Recommendations

- **Model Enhancement:** Experiment with transformer-based models fine-tuned for finance, such as **FinBERT** or **BloombergGPT**, to better capture domain-specific sentiment.
- **Data Expansion:** Increase data coverage across multiple years and companies to reduce overfitting and improve generalization.
- **Feature Integration:** Combine sentiment features with **technical indicators** (e.g., moving averages, RSI) and **fundamental metrics** for multi-factor modeling.
- **Real-Time Application:** Integrate live financial feeds with real-time sentiment scoring for dynamic investment strategy insights.

---

## 🚀 Future Work

- Implement **transfer learning** with larger transformer architectures.  
- Automate **real-time news ingestion and sentiment scoring**.  
- Explore **ensemble models** combining different embeddings for robustness.  
- Develop a dashboard for **interactive sentiment tracking and visualization**.

---

## 🧰 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.10+ |
| **NLP** | NLTK, Gensim (Word2Vec, GloVe), SentenceTransformers |
| **ML Models** | scikit-learn (GradientBoostingClassifier) |
| **Visualization** | Matplotlib, Seaborn |
| **Data Handling** | Pandas, NumPy |

---

## 📈 Results Summary

| Embedding Type | Train F1 | Validation F1 | Observation |
|----------------|-----------|----------------|--------------|
| Word2Vec | High | Moderate | Overfit |
| GloVe | Highest | Best balance | Selected final model |
| SentenceTransformer | Moderate | Inconsistent | Needs more data |

---

## 📚 References

- [GloVe: Global Vectors for Word Representation (Stanford NLP)](https://nlp.stanford.edu/projects/glove/)
- [Word2Vec: Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)
- [SentenceTransformers](https://www.sbert.net/)
- [Yahoo Finance API](https://finance.yahoo.com/)
- [Kaggle Financial News Datasets](https://www.kaggle.com/)

---

## 🗂️ License

This project is intended for **academic and educational purposes only**.  
Data sources are public and used under fair research and educational use.

---

## ⭐ Acknowledgements

- **Gensim** for word embedding implementation  
- **scikit-learn** for model evaluation utilities  
- **SentenceTransformers** for contextual embeddings  

---


