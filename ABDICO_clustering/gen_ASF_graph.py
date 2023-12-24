from bertopic import BERTopic
from hdbscan import HDBSCAN
import pandas as pd
import networkx as nx

components = ['Attribute','Deontic','Object']
result = pd.read_csv('main.csv', usecols=components)
result.dropna(how='all',inplace=True)
result = result[result['Deontic'].isin['should','may','can','must','could']]

for component in ['Attribute','Object']:
    entries = result[component].tolist()

    hdbscan_model = HDBSCAN(metric='euclidean', cluster_selection_method='leaf', min_cluster_size=2, prediction_data=True)
    topic_model = BERTopic(top_n_words = 3,hdbscan_model=hdbscan_model,nr_topics='auto')
    topic_model.hdbscan_model.gen_min_span_tree=True
    topic_model.umap_model.random_state= 0 ##set seed to enable reproduction of clustering

    topic_model.fit(entries)
    freq = topic_model.get_topic_info()
    result[component + '_group'] = topic_model.transform(entries)[0]
    result[component + '_group'] = result[component + '_group'].apply(lambda x : freq[freq['Topic']==x]['Name'].to_list()[0])


G = nx.MultiDiGraph()

for _, row in result.iterrows():
    G.add_edge_from([row.Attribute_inf, row.Object_inf])

nx.draw(G)