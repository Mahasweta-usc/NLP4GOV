# Semantic Search Code Notebook

## Theoretical Framework

Semantic search is a powerful technique used to map informal statements to formal rules, enabling direct comparisons of "rules-in-use" to "rules-in-form." This analysis addresses critical issues related to policy implementation, policy internalization, and emergent norms. This notebook leverages advanced natural language processing to perform semantic searches, linking a "needle" (a single institutional statement) with a "haystack" (a large corpus of formal or informal institutional statements). The goal is to identify informal statements that are most similar to the formal policy statement, based on numerical similarities.

The primary use case demonstrated in this notebook involves taking a formal policy statement as the "needle" and searching through a vast collection of email sentences (the "haystack") for statements related to the policy. This kind of analysis can have broader applications, allowing comparisons between "rules-in-use" and "rules-in-form" in various contexts.

## Notebook Usage

**Getting Started:** To use this notebook, start by running the provided code cells. You can either work with the sample datasets provided or upload your own datasets. The notebook is designed to be flexible and adaptable to different policy analysis scenarios.

**Data Preparation:** The code reads policy data from CSV files, creating two dataframes, `query` and `search_base`, to represent the policy statement and the corpus of sentences to search within. Ensure that your datasets adhere to the expected structure, with columns named "document" and "reply."

**Analysis and Visualization:** The notebook performs a series of steps, including BM25Okapi scoring to filter the most relevant sentences, extracting sentence embeddings, and ranking matches with the query by cosine distance. This process allows the identification of sentences in the "haystack" that are most similar to the "needle." The results are visualized and can be further analyzed.

**Customization:** Advanced users can modify the notebook to suit specific research objectives. Customization options include altering algorithms, settings, and visualizations to align with your analysis needs.

## Dependencies

The notebook relies on various essential Python libraries and external models, including `pandas`, `nltk`, `scikit-learn`, `sentence_transformers`, and `rank_bm25`. These libraries are fundamental for text processing, BM25 scoring, and semantic similarity calculations. The Sentence Transformer model, "Jainam/freeflow-biencoder," plays a crucial role in generating sentence embeddings.

## Explore Further

This notebook provides a platform to explore advanced concepts in natural language processing and policy analysis:

- **Semantic Search:** Deepen your understanding of semantic search techniques, which facilitate mapping informal statements to formal rules, allowing for direct comparisons and analysis.

## Technical Concepts and Libraries

### BM25 Okapi Scoring

**BM25 Okapi** is a ranking function used for information retrieval, often applied to search engines and text-based recommendation systems. It is an improved version of the **BM25** algorithm, which stands for **Best Matching 25**. Here's a more detailed breakdown:

- **BM25**: This ranking function assigns scores to documents based on their relevance to a query. It considers factors like term frequency, document length, and document frequency to rank documents. BM25 is widely used in search engines and information retrieval systems because it balances precision and recall effectively.

- **BM25 Okapi**: The "Okapi" part of the name refers to the Okapi BM25 model, which was developed at the City University of London. BM25 Okapi extends the BM25 algorithm with additional parameters and refinements to improve its performance.

### Libraries

The notebook relies on several Python libraries for data processing, text analysis, and natural language understanding:

- **Pandas**: [Pandas](https://pandas.pydata.org/) is a versatile data manipulation library that provides data structures for efficiently working with structured data, including dataframes used for data representation and analysis.

- **NLTK (Natural Language Toolkit)**: [NLTK](https://www.nltk.org/) is a powerful library for working with human language data. It provides tools for text analysis, tokenization, stemming, and more. In this notebook, NLTK is used for text preprocessing.

- **Scikit-learn**: [Scikit-learn](https://scikit-learn.org/stable/index.html) is a machine learning library that includes tools for data mining and data analysis. It provides various utilities for feature extraction and text processing. In this notebook, Scikit-learn is used for stop word removal and other text processing tasks.

- **Sentence Transformers**: [Sentence Transformers](https://www.sbert.net/) is a library that focuses on generating numerical representations of sentences, often referred to as embeddings. It uses transformer-based models to create these embeddings, making it ideal for tasks like semantic similarity calculations. The library is used in this notebook to encode sentences and measure their similarity.

- **Rank BM25**: [Rank BM25](https://pypi.org/project/rank-bm25/) is a Python library for BM25 and BM25 Okapi ranking. It's a useful tool for ranking documents based on their relevance to a query. In this notebook, Rank BM25 is used for initial scoring to filter the most relevant sentences.

- **Word Embeddings:** Explore word embeddings and their role in converting text data into numerical representations that enable semantic similarity comparisons.

- **Customization:** Take full advantage of the notebook's flexibility and adapt it to specific research or policy analysis scenarios.

## In and Out Information

**Input Data:** The notebook expects two primary datasets - one containing a formal policy statement ("needle") and the other consisting of a corpus of sentences ("haystack") from which the relevant statements will be extracted. The expected columns are "document" and "reply." These datasets are crucial for the search and analysis.

**Output Data:** The notebook generates valuable outputs, including a dataframe of the most relevant statements from the "haystack," along with their similarity scores. Additionally, it provides a downloadable CSV file containing the sorted results for further analysis and exploration.
