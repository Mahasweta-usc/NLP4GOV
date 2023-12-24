from bertopic import BERTopic
from hdbscan import HDBSCAN
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(18, 15), dpi=300)

components = ['Attribute', 'Deontic', 'Object']
result = pd.read_csv('main.csv', usecols=components)
result.replace("", np.nan, inplace=True)
result.dropna(how='any', inplace=True)
result = result[result['Deontic'].isin(['should', 'may', 'can', 'must', 'could'])]

for component in ['Attribute', 'Object']:
    entries = result[component].tolist()
    hdbscan_model = HDBSCAN(metric='euclidean', cluster_selection_method='eom', min_cluster_size=2,
                            prediction_data=True)
    topic_model = BERTopic(top_n_words=3, hdbscan_model=hdbscan_model)
    topic_model.hdbscan_model.gen_min_span_tree = True
    topic_model.umap_model.random_state = 0  ##set seed to enable reproduction of clustering

    topic_model.fit(entries)
    freq = topic_model.get_topic_info()
    result[component + '_group'] = topic_model.transform(entries)[0]
    # remove outliers
    result[component + '_group'].replace("-1", np.nan);
    result.dropna(inplace=True)
    result[component + '_group'] = result[component + '_group'].apply(
        lambda x: freq[freq['Topic'] == x]['Name'].to_list()[0])

G = nx.MultiDiGraph()

for _, row in result.iterrows():
    G.add_edge(row.Attribute_group, row.Object_group)

pos = nx.random_layout(G)
nx.draw_networkx_nodes(G, pos, node_color='r', node_size=100, alpha=1)
ax = plt.gca()

for e in G.edges:
    ax.annotate("",
                xy=pos[e[0]], xycoords='data',
                xytext=pos[e[1]], textcoords='data',
                arrowprops=dict(arrowstyle="->", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3 * e[2])
                                                                       ),
                                ),
                )

for node in G.nodes:
    ax.text(pos[node][0], pos[node][1], node, fontsize=16)

plt.axis('off')
plt.show()