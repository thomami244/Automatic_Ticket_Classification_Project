# Automatic Ticket Classification

Welcome to this **Case Study** on **Automatic Ticket Classification**. In this project, we build a model to automatically classify customer complaints into relevant categories, thereby helping a financial company streamline its support processes.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Business Goal](#business-goal)
3. [Dataset](#dataset)
4. [Approach & Methodology](#approach--methodology)
   1. [Data Loading](#data-loading)
   2. [Text Preprocessing](#text-preprocessing)
   3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   4. [Feature Extraction](#feature-extraction)
   5. [Topic Modelling (NMF)](#topic-modelling-nmf)
   6. [Model Building (Supervised Learning)](#model-building-supervised-learning)
   7. [Model Training & Evaluation](#model-training--evaluation)
   8. [Model Inference](#model-inference)
5. [Results](#results)
   1. [Logistic Regression](#1-logistic-regression)
   2. [Decision Tree](#2-decision-tree)
   3. [Random-Forest](#3-random-forest)
6. [Conclusion & Recommendation](#conclusion--recommendation)
7. [How to Use this Repository](#how-to-use-this-repository)
8. [License](#license)

---

## Introduction

**Problem Statement**  
A financial company receives thousands of customer complaint tickets every day, covering various products/services such as:
- Credit cards / Prepaid cards
- Bank account services
- Theft/Dispute reporting
- Mortgages/Loans
- Others

Traditionally, support teams manually evaluate each complaint and allocate it to the respective department. This method is time-consuming, prone to errors, and does not scale well. Our goal is to **automate** this process using **Natural Language Processing (NLP)** techniques to improve efficiency and customer satisfaction.

---

## Business Goal

1. **Automate customer complaint classification** based on product/service categories.
2. **Segregate tickets** into relevant categories for quicker resolution.
3. **Leverage topic modelling** (NMF) to detect recurring words and patterns in complaint text.
4. **Build supervised models** (e.g., Logistic Regression, Decision Tree, Random Forest) using the labelled data for final classification.

Ultimately, this solution will help the company:
- Reduce manual work and **operational costs**.
- Resolve complaints **faster** and thus improve **customer satisfaction**.
- Gain insights into recurring complaint patterns to **continually improve** their services.

---

## Dataset

- The dataset is provided in **.json** format.
- Consists of **78,313 customer complaints** with **22 features**.
- After loading and pre-processing, the data is converted into a **pandas DataFrame** for analysis.
- The complaints are **unstructured text data** that need extensive cleaning and vectorization.

---

## Approach & Methodology

Below is the high-level methodology we followed:

### Data Loading
- Loaded the `.json` dataset into a pandas DataFrame.
- Inspected the dataset to identify relevant features such as complaint text.

### Text Preprocessing
1. **Tokenization** of complaint text.
2. **Stopword removal** (e.g., "the", "and", etc.).
3. **Lemmatization** or **Stemming** to convert words to their base forms.
4. **Lowercasing** to ensure uniform text.

### Exploratory Data Analysis (EDA)
- Reviewed **word frequency**, **common bigrams/trigrams**, and **length distributions**.
- Analyzed distribution of complaint topics (once labelled) to understand major areas of concern.

### Feature Extraction
- Utilized **TF-IDF** or **Count Vectorizer** to transform text into numeric features.
- Created word vectors for further analysis in topic modelling and classification.

### Topic Modelling (NMF)
- Used **Non-negative Matrix Factorization (NMF)** to discover **latent topics** in the unlabelled data.
- Mapped each complaint to one of the five clusters identified:
  1. **Credit card / Prepaid card**
  2. **Bank account services**
  3. **Theft/Dispute reporting**
  4. **Mortgages/Loans**
  5. **Others**

### Model Building (Supervised Learning)
- Once each complaint was assigned a label via topic modelling, these labels formed our **training dataset**.
- Tested at least three models:
  1. [Logistic Regression](#1-logistic-regression)
  2. [Decision Tree](#2-decision-tree)
  3. [Random Forest](#3-random-forest)

### Model Training & Evaluation
- **Hyperparameter tuning** was done to optimize model performance.
- Evaluation metrics included:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**

### Model Inference
- Validated the **best performing model** on a test set to ensure generalization.
- Final model can now be used to classify **new, incoming** customer complaints.

---

## Results

### 1. Logistic Regression

**Accuracy**: 0.92

**Classification Report**:

               accuracy                           0.92      4215
              macro avg       0.92      0.90      0.91      4215
           weighted avg       0.92      0.92      0.92      4215


### 2. Decision Tree

**Accuracy**: 0.80

**Classification Report**:

               accuracy                           0.80      4215
              macro avg       0.80      0.79      0.79      4215
           weighted avg       0.80      0.80      0.80      4215


### 3. Random Forest

**Accuracy**: 0.80

**Classification Report**:

               accuracy                           0.80      4215
              macro avg       0.84      0.75      0.77      4215
           weighted avg       0.82      0.80      0.80      4215



---

## Conclusion & Recommendation

Based on the evaluation:
- **Logistic Regression** consistently shows the highest accuracy (0.92).
- It also has the highest weighted average for **precision**, **recall**, and **F1-score**.
  
Hence, **Logistic Regression** is recommended for the final deployment because it offers the best performance across all evaluation metrics.

---






