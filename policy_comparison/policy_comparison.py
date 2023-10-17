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
data = '/content/'
file_names = ['db1.csv','db2.csv']

for file_name in file_names:
    file_path = os.path.join(data, file_name)
    if os.path.isfile(file_path):
        if "db1.csv" in file_name:
            db1 = pd.read_csv(file_path, usecols=[column_name])
        elif "db2.csv" in file_name:
            db2 = pd.read_csv(file_path, usecols=[column_name])

db1 = db1.dropna()
db1 = db1.reset_index(drop=True)

db2 = db2.dropna()
db2 = db2.reset_index(drop=True)

word_embedding_model = SentenceTransformer("all-mpnet-base-v2")

class policy_comparison:
    def __init__(self, agent=None):
        self.agent = agent

    def sentence_embeddings_encode(self, word_embedding_model, data):
        return word_embedding_model.encode(data)
    
    def show_db(self):
        print("Policy Database 1:", db1.head(5))
        print("\n\n")
        print("Policy Database 2:", db2.head(5))
        return db1, db2
    
    def show_results(self, search_results):
        # Retrieve and store the results in the DataFrame
        results_df = pd.DataFrame(columns=["Policy Database 1", "Policy Database 2", "Similarity Score"])

        seen_pairs = set()
        for i, query_result in enumerate(search_results):
            for j, result in enumerate(query_result):
                corpus_id = result['corpus_id']
                score = result['score']
                # Access the corresponding sentences from db1 and db2
                sentence1 = db1.loc[i, 'Raw Institutional Statement']
                sentence2 = db2.loc[corpus_id, 'Raw Institutional Statement']

                # Sort the sentences alphabetically to eliminate duplicates
                pair_key = tuple(sorted([sentence1, sentence2]))
                
                # Check if this pair has been seen before
                if pair_key not in seen_pairs:
                    results_df.loc[len(results_df)] = [sentence1, sentence2, score]
                    seen_pairs.add(pair_key)
                    
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

        fig, axs = plt.subplots(2)
        plt.subplots_adjust(bottom=0.15)

        for item, db in enumerate([db1, db2]):
            sentences_list = db["Raw Institutional Statement"].tolist()
            combined_text = ' '.join(sentences_list)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
            ax[item].imshow(wordcloud, interpolation='bilinear')
            title = "Policy Database 1" if not item else "Policy Database 2"

            ax[item].set_title(title, pad=20)

        fig.suptitle("Word Cloud of Policy Sets")
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
