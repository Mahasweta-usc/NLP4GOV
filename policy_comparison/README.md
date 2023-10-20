# Policy Comparison Code Notebook

## Introduction

Welcome to the Policy Comparison Code Notebook! This notebook leverages advanced natural language processing techniques to analyze and compare policy statements or rules from two different sets. It is a powerful tool for policy scholars, institutions, and analysts to compare rules within or between institutions. This README provides a detailed overview of the methods and processes implemented in the notebook.

### Sentence Embeddings

- The notebook employs sentence embeddings, which are numerical representations of each sentence in the provided datasets. These embeddings are generated using a Sentence Transformer model, specifically the "Jainam/freeflow-biencoder."

- Each sentence is represented as a vector in a 768-dimensional semantic space. The neural network-based model assigns coordinates to sentences in this space. The remarkable aspect is that sentences with similar coordinates are semantically similar, even when accounting for context.

### All-to-All Comparison

- An "all-to-all" comparison is performed, meaning that every statement in the first dataset is compared to every statement in the second dataset. This comprehensive approach ensures that all possible similarities are explored.

- For instance, if the first dataset contains 10 statements and the second dataset contains 15, this results in a total of 150 pairwise comparisons, creating a detailed picture of the similarity between policy statements.

### Numerical Similarity Scores

- The notebook calculates numerical similarity scores for each pair of statements. These scores quantify the similarity or dissimilarity of pairs in absolute terms, helping users gauge the degree of similarity between policy statements.

- Higher scores indicate greater similarity, while lower scores suggest dissimilarity.

## Notebook Usage

### Getting Started

1. **Run the Notebook:** Execute the provided code cells to run the analysis. The notebook can be run with example datasets or with your own data.

2. **Upload Your Data:** For your specific analysis, you can upload your own datasets. Ensure that you adapt the notebook to your data format, as described below.

### Data Preparation

- The code reads data from CSV files. Two dataframes, `db1` and `db2`, are created, representing two datasets of policy statements.

- Ensure that your datasets are formatted correctly, with a column named "Raw Institutional Statement" containing the policy statements.

### Analysis and Visualization

- The notebook performs the following tasks:

  - Generates sentence embeddings for both datasets using the Sentence Transformer model.
  - Conducts an "all-to-all" comparison of the policy statements.
  - Calculates numerical similarity scores for each pair of statements.
  - Visualizes the results through distribution plots, word clouds, and a 3D scatter plot.

### Customization

- The notebook is highly customizable for users with programming experience. You can modify settings, algorithms, and visualizations to meet your specific research or policy analysis needs.

## Dependencies

- The notebook uses several Python libraries, including `pandas`, `seaborn`, `matplotlib`, `wordcloud`, `mpl_toolkits`, and `sentence_transformers`. Make sure you have these dependencies installed for the notebook to work correctly.

## Explore Further

- This notebook offers an opportunity to delve into the world of semantic similarity, natural language processing, and advanced analysis methods. Users can adapt the notebook, explore advanced concepts, and customize it for specific research purposes.

## In and Out Information

- **Input Data:** The notebook requires two sets of policy sentences in CSV format, each with a column named "Raw Institutional Statement."

- **Output Data:** The notebook provides the following results:
  - A dataframe with statements most similar to each other.
  - Numerical similarity scores for each pair of statements.
  - A distribution plot illustrating the distribution of similarity scores.
  - A word cloud summarizing significant terms or themes.
  - A 3D scatter plot showing relationships between data points.

