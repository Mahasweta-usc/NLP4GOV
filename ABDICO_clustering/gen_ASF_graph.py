from bertopic import BERTopic
from hdbscan import HDBSCAN
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import sklearn
stopwords = list(sklearn.feature_extraction.text.ENGLISH_STOP_WORDS)
figure(figsize=(18, 15), dpi=300)


import stanza
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse',use_gpu=True)

def topic_name(x):
    org = x
    x = x.split('_')[1:]
    x = [elem for elem in x if elem and (elem not in stopwords)]
    x = set([[word.lemma for sent in nlp(elem).sentences for word in sent.words][0] for elem in x])
    return ",\n".join(x)

deontic_map = {"must": 'black',
               "should": "blue",
               "can": 'limegreen',
               "may": 'limegreen',
               "might": "limegreen",
               "could": 'limegreen',
               "other": "gray"}

components = ['Attribute', 'Deontic', 'Object']
result = pd.read_csv('main.csv', usecols=components)
result.replace("", np.nan, inplace=True)
result.dropna(subset=["Attribute", "Object"], how='any', inplace=True)

#replace first person
for col in ["Attribute", "Object"]:
    result[col] = result[col].apply(lambda x: x.replace(" we ", " asf "))
    result[col] = result[col].apply(lambda x: x.replace("your ", "project "))
    result[col] = result[col].apply(lambda x: x.replace(" our", " asf"))
    result[col] = result[col].apply(lambda x: x.replace("you ", "project "))
    result[col] = result[col].apply(lambda x: x.replace(" us ", " asf  "))

    result[col] = result[col].replace("we","asf")
    result[col] = result[col].replace("our", "asf")
    result[col] = result[col].replace("your", "project")
    result[col] = result[col].replace("you", "project")
    result[col] = result[col].replace("us", "asf")

result.fillna("",inplace=True)
result['Deontic'] = result['Deontic'].apply(lambda x : x if x in deontic_map else "other")

entries = result['Attribute'].tolist()
entries.extend(result['Object'].tolist())
hdbscan_model = HDBSCAN(metric='euclidean', cluster_selection_method='eom',min_cluster_size=5,
                        prediction_data=True)
topic_model = BERTopic(top_n_words=3, hdbscan_model=hdbscan_model)
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

for _, row in result.iterrows():
    # try:
    #     G[row.Attribute_group][row.Object_group]['weight'] += 1
    # except Exception as exp:
    #     print(exp)
    G.add_edge(row.Attribute_group, row.Object_group, color=row.Deontic)

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color='r', node_size=100, alpha=1)
ax = plt.gca()

for (u,v,attrib_dict) in list(G.edges.data()):
    style = str(0.3 + 0.3*np.random.rand())
    ax.annotate("",
                xy=pos[u], xycoords='data',
                xytext=pos[v], textcoords='data',
                arrowprops=dict(arrowstyle="-|>,head_length=.8,head_width=.4", color=attrib_dict['color'],
                shrinkA=5, shrinkB=5,
                patchA=None, patchB=None,
                connectionstyle=f"arc3,rad={style}"
                                ),
                )

for node in G.nodes:
    ax.text(pos[node][0]+.005, pos[node][1]+.005, node, fontsize=16, bbox=dict(facecolor='lemonchiffon', edgecolor='black'))

plt.axis('off')
plt.savefig("ASF_Graph.jpg",dpi=300)