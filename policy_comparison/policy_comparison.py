import os, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, HTML
from sentence_transformers import SentenceTransformer, util

#read files
column_name = "Raw Institutional Statement"
file_names = os.listdir('/content/IG-SRL/policy_comparison/data')
db1 = pd.read_csv(file_names[0], usecols=[column_name])
db2 = pd.read_csv(file_names[1], usecols=[column_name])

db1 = db1.dropna()
db1 = db1.reset_index(drop=True)

db2 = db2.dropna()
db2 = db2.reset_index(drop=True)

print(db1)
print(db2)

word_embedding_model = SentenceTransformer("Jainam/freeflow-biencoder")

results_df = pd.DataFrame(columns=["Camden Food Security", "Connecticut Food Policy", "Similarity Score"])

class policy_comparison:
    def __init__(self, agent=None):
        self.agent = agent

    def sentence_embeddings_encode(word_embedding_model, data):
        return word_embedding_model.encode(data)
    
    def show_results(self, search_results):
        # Retrieve and store the results in the DataFrame
        for i, query_result in enumerate(search_results):
            for j, result in enumerate(query_result):
                corpus_id = result['corpus_id']
                score = result['score']
                # Access the corresponding sentences from db1 and db2
                sentence1 = db1.loc[i, 'Raw Institutional Statement']
                sentence2 = db2.loc[corpus_id, 'Raw Institutional Statement']
                results_df.loc[len(results_df)] = [sentence1, sentence2, score]
        results_df = results_df.sort_values(by='Similarity Score', ascending=False)
        display(HTML(results_df.to_html()))
        return results_df

    def plot_similarity_frequency(self, results_df):
        scores = results_df["Similarity Score"]
        plt.figure(figsize=(10, 6))
        sns.kdeplot(scores, color='blue', fill=True, bw_method=0.05)
        plt.xlabel('Similarity Score')
        plt.ylabel('Probability Density')
        plt.title('Similarity Score PDF')
        plt.show()
    
    def plot_WordCloud(self, results_df):
        sentences_list = results_df["Camden Food Security"].tolist() + results_df["Connecticut Food Policy"].tolist()
        combined_text = ' '.join(sentences_list)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Word Cloud of Sentences")
        plt.show()
    
    def plot_Similarity_Scores_3D(self, results_df):
        sentence1_ids = results_df.index
        sentence2_ids = results_df.index
        scores = results_df["Similarity Score"]

        fig = plt.figure(figsize=(15, 13))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sentence1_ids, sentence2_ids, scores, c=scores, cmap='coolwarm', marker='o')
        ax.set_xlabel('Sentence 1 Embedding ID')
        ax.set_ylabel('Sentence 2 Embedding ID')
        ax.set_zlabel('Similarity Score')
        ax.set_title('Similarity Score Plot')