# IPLN Course Repository

This repository demonstrates the application of various Natural Language Processing (NLP) techniques through tasks learned in the IPLN Course at Fing-UdelaR. The included Jupyter notebooks provide step-by-step examples of sentiment analysis and syntactic parsing, using Python-based tools and libraries.

## Introduction

### Overview  
The objective is to conduct various experiments to represent and classify tweets based on their 3-class polarity sentiment (positive, neutral, and negative). These experiments utilize the same corpus from the [TASS 2020](http://www.sepln.org/workshops/tass/) competition (IberLEF - SEPLN), which focuses on Spanish sentiment analysis. 

Different machine learning models are compared using their Macro-F1 scores on the test set. Results are also benchmarked against state-of-the-art performance using the [pysentimiento](https://github.com/pysentimiento/pysentimiento) library for Spanish sentiment analysis.

## Contents

### 1. **SyntacticAnalysis.ipynb**  
This notebook focuses on syntactic analysis, exploring the structure of sentences to understand grammatical relationships between words. Key aspects include:  
- Part-of-speech tagging to identify word types (nouns, verbs, etc.).  
- Dependency parsing to map relationships between words in a sentence.  
- Constituency parsing to break down sentences into sub-phrases or constituents.  
- Visualizing syntactic structures.

### 2. **SentimentAnalysis.ipynb**  
This notebook demonstrates the process of sentiment analysis on textual data. It involves:  
- Data preprocessing and cleaning.  
- Feature extraction using NLP techniques.  
- Training machine learning models for text classification:
  - Support Vector Machines (SVM)
  - Logistic Regression and Multi-Layer Perceptrons (MLPs)
  - LSTM Neural Networks
- Evaluating model performance and visualizing results.  

The goal is to classify text based on its sentiment (e.g., positive, neutral, negative), providing insights into the underlying emotional tone of the data.

