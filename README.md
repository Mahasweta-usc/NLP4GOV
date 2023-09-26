
# Contents

1. Recommended pipelines and applications
2. Navigating Colaboratory 
3. Tasks Overview

# Recommended pipelines and applications

The repository is a joint effort led by <INSERT Labs/Organizations involved>. We present an extensive, curated collection of  functionalities and tasks in natural language processing, adapted to aid collective action research at scale. 

Our Github currently hosts 5 (more soon!) versatile end to end applications to process raw policy corpus and extract meaningful features for research and analysis. Examples include but not limited to:

* **Identify community players, strength of regulation and interactions:** Preprocess documents > Extract ABDICO components > Network Analysis
* **Policy diffusion/adaption over time:** Preprocess policy documents > Compare policies with interviews/conversations
* **Compare Institutions:** Preprocess policy documents from different communities > Find most similar rules between two communities
* **Power dynamics in communities:** Preprocess policy documents > Extract ABDICO components > cluster actors > Analyze inter group leverage

# Navigating Colaboratory

Colaboratory or Colab is a service provided by Google, which allows users to run and develop notebook applications while leveraging their infrastructure at very little cost.
## Overview of Colab subscription plans:

* Free : Nvidia T4 GPU
* Pro ($9.99/month) : Nvidia T4/ V100/ A100 + High RAM CPU. Check website for unit rate charges for different GPUs and their features.

For most notebooks here however, users should not require premium subscriptions. A free T4 GPU generally suffices, unless the specific use case involves high volumes (hundreds of MBs to several GBs) of text data and necessitates more compute/speed.

## Getting Started

1. Download this repository (compressed zip) to your local system
      ![img.png](images/img1.png)
2. Extract downloaded file. Notebook applications end in a '.ipynb' extension. 
![img.png](images/img2.png)
3. Go to https://colab.research.google.com/. Upload your selected notebook from the repo as shown
![img.png](images/img3.png)
4. Set notebook backend. Select Runtime (Upper left header). Make sure you are opting for a GPU and using Python 3.
![img.png](images/img4.png)
5. Run first cell of each notebook for installations and package imports
![img.png](images/img5.png)
6. Follow inline instructions to run the remaining notebook cells one by one
![img.png](images/img6.png)
7. Download the final file with the results (Generally "main.csv") from right hand directory panel.
![img.png](images/img7.png)