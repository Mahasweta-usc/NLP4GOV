from bertopic import BERTopic
from hdbscan import HDBSCAN
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.lines import Line2D
import sklearn
from operator import add

stopwords = list(sklearn.feature_extraction.text.ENGLISH_STOP_WORDS)
np.random.seed(0)

import torch
torch.manual_seed(0)

import stanza

stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', use_gpu=True)


def topic_name(x):
    org = x
    x = x.split('_')[1:]
    x = set([" ".join([word.lemma for sent in nlp(elem).sentences for word in sent.words]) for elem in x])
    words = []
    for elem in x:
        elem = set(elem.split()) - set(stopwords)
        if elem: words.append("\n".join(elem))
    return ",\n".join(set(words))


deontic_map = {"must": 'black',
               "should": "mediumpurple",
               "will not" : "red",
               "shall not" : "red",
               "should not" : "red",
               "not" : "red",
               "can": 'green',
               "may": 'green',
               "might": "green",
               "could": 'green',
               "other": "green",
               'may not': 'red',
               'can not': 'red',
               'must not': 'red'}

components = ['Attribute', 'Deontic', 'Object']
result = pd.read_csv('main.csv', usecols=components)
result.replace("", np.nan, inplace=True)
result.dropna(subset=["Attribute", "Object"], how='any', inplace=True)

# replace first person
for col in ["Attribute", "Object"]:
    result[col] = result[col].apply(lambda x: x.replace(" we ", " asf "))
    result[col] = result[col].apply(lambda x: x.replace("your ", "project "))
    result[col] = result[col].apply(lambda x: x.replace(" our", " asf"))
    result[col] = result[col].apply(lambda x: x.replace("you ", "project "))
    result[col] = result[col].apply(lambda x: x.replace(" us ", " asf  "))

    result[col] = result[col].replace("we", "asf")
    result[col] = result[col].replace("our", "asf")
    result[col] = result[col].replace("your", "project")
    result[col] = result[col].replace("you", "project")
    result[col] = result[col].replace("us", "asf")

result.fillna("", inplace=True)
result['Deontic'] = result['Deontic'].apply(lambda x: x if x in deontic_map else "other")

entries = result['Attribute'].tolist()
entries.extend(result['Object'].tolist())
hdbscan_model = HDBSCAN(metric='euclidean', cluster_selection_method='eom', min_cluster_size=15, min_samples=5,
                        prediction_data=True)
topic_model = BERTopic(top_n_words=3, hdbscan_model=hdbscan_model, nr_topics='auto', n_gram_range=(1, 2))
topic_model.hdbscan_model.gen_min_span_tree = True
topic_model.umap_model.random_state = 0  ##set seed to enable reproduction of clustering

topic_model.fit(entries)
freq = topic_model.get_topic_info()

for component in ['Attribute', 'Object']:
    entries = result[component].tolist()
    result[component + '_group'] = topic_model.transform(entries)[0]
    # # remove outliers
    # result = result[result[component + '_group'] != -1]
    result[component + '_group'] = result[component + '_group'].apply(
        lambda x: freq[freq['Topic'] == x]['Name'].to_list()[0])
    result[component + '_group'] = result[component + '_group'].apply(lambda x: topic_name(x))

result['Deontic'] = result['Deontic'].apply(lambda x: deontic_map[x])

G = nx.MultiDiGraph()

# for _, row in result.iterrows():
#     # try:
#     #     G[row.Attribute_group][row.Object_group]['weight'] += 1
#     # except Exception as exp:
#     #     print(exp)
#     G.add_edge(row.Attribute_group, row.Object_group, color=row.Deontic)

SNR_map = {'green': 'Strategies',
           'mediumpurple': 'Recommended Norms/Requirements : Should', 'black': "Binding Norms/Requirements : Must",
           'red': 'Restrictions'}

for idx, row in result.iterrows():
    # G.add_edge(row.Attribute_group, row.Object_group, weight=1, color=row.Deontic, key=idx)
    data = G.get_edge_data(row.Attribute_group, row.Object_group, default ={})
    try:
        data = [v for k,v in data.items() if v["color"] == row.Deontic][0]
        # we added this one before, just increase the weight by one
        G.remove_edge(row.Attribute_group, row.Object_group, key=row.Deontic)
        G.add_edge(row.Attribute_group, row.Object_group, color=row.Deontic, weight=data['weight'] + 1, key=row.Deontic)
    except Exception as exp:
        G.add_edge(row.Attribute_group, row.Object_group, weight = 1, color=row.Deontic, key=row.Deontic)

pos = nx.spring_layout(G, k=0.0)
# ax = plt.gca()
fig, axes = plt.subplots(2, 2, figsize=(20, 30))
axes = axes.flatten()

for idx, shade in enumerate((SNR_map.keys())):
    edges = [(u, v, k) for u, v, k in G.edges if G[u][v][k]['color'] == shade]
    weights = [(np.log2(G[u][v][k]['weight']) + 1)*1.5 for u, v, k in edges]
    nodes = [];
    for u, v, x in edges:
        nodes.extend([u, v])

    print("nodes: ", set(nodes))

    new_G = nx.MultiDiGraph()
    new_G.add_nodes_from(nodes)

    # pos = nx.kamada_kawai_layout(new_G)
    # out_track = {}
    # outedge = [edge[0] for edge in edges]
    # for node in set(nodes):
    #     count = {node: outedge.count(node)}
    #     out_track.update(count)
    # out_track = {k: v for k, v in sorted(out_track.items(), key=lambda item: item[1], reverse=True)}
    #
    # in_track = {}
    # inedge = [edge[1] for edge in edges]
    # for node in set(nodes):
    #     count = {node: inedge.count(node)}
    #     in_track.update(count)
    # in_track = {k: v for k, v in sorted(in_track.items(), key=lambda item: item[1], reverse=True)}

    # print(shade, "in_degree: ", in_track)
    # print("out_degree", out_track)

    axes[idx].set_title(SNR_map[shade], fontsize=32, fontweight='heavy')
    #draw node edges
    # nodes = nx.draw_networkx_nodes(G, pos, node_color='lemonchiffon', node_size=40000)
    # nx.draw_networkx_labels(G, pos, font_size=25, alpha=1 , font_weight='bold')
    # nodes.set_edgecolor('r')
    #
    # nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=shade, width=weights,
    #                  connectionstyle=f"arc3,rad=-0.5",
    #                  arrowstyle=f"-|>,head_length=1.5,head_width=1.2", ax=axes[idx])  #

    nx.draw_networkx(new_G, pos, node_color='lemonchiffon', nodelist=set(nodes), font_size=18, edgelist=edges,
                           edge_color=shade, width=weights,
                           node_size=20000, alpha=1, with_labels=True, font_weight='bold',
                           connectionstyle=f"arc3,rad=-0.5",
                           arrowstyle=f"-|>,head_length=1.2,head_width=1", ax=axes[idx])  #

fig.tight_layout()

# edge_labels = nx.get_edge_attributes(G,'weight')
# nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)

# for (u,v,attrib_dict) in list(G.edges.data()):
#     radius = str(0.1 + 0.5*np.random.rand())
#     length = str(0.5 + 0.05*attrib_dict['weight'])
#     width = str(0.3 + 0.1**attrib_dict['weight'])
#     style = f"-|>,head_length={length},head_width={width}"
#     ax.annotate("",
#                 xy=list(map(add, pos[u], [0.05,-0.05])), xycoords='data',
#                 xytext=list(map(add, pos[v], [-0.05,0.05])), textcoords='data',
#                 arrowprops=dict(arrowstyle=style, color=attrib_dict['color'],lw=1.5*attrib_dict['weight'],
#                 shrinkA=5, shrinkB=5,
#                 patchA=None, patchB=None,
#                 connectionstyle=f"arc3,rad={radius}"
#                                 ),
#                 )

# for node in G.nodes:
#     ax.text(pos[node][0]+.01, pos[node][1]+.01, node, fontsize=16, bbox=dict(facecolor='lemonchiffon', edgecolor='black'))


# custom_lines = [Line2D([0], [0], color='lightblue', lw=4),
#                 Line2D([0], [0], color='green', lw=4),
#                 Line2D([0], [0], color='mediumpurple', lw=4),
#                 Line2D([0], [0], color='black', lw=4)]


# ax.legend(custom_lines, ['Strategies', 'May/Can', 'Should', "Must"], ncol=4, loc="upper right", prop={'size': 16})

# plt.axis('off')
plt.savefig("ASF_Graph.jpg", dpi=300)