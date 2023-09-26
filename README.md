
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

# Tasks Overview

## General Guidelines

* All input files to notebooks must be .csv. If your data is in any other formal (xls/json), please make sure to convert appropriately before running a notebook.
* Check comments in each notebook to understand how input data should be configured, i.e. upload file name, table headers/input data fields/output data fields.
* When coding raw policy statements for some notebooks (e.g. ABDICO_parsing.ipynb, policies_compares, etc ), for interpretability as well as best results, it's recommended to present data in self contained statements with most IG components present than bullets.

      E.g. "The Board should 1. conduct a vote every 3 years 2. review proposal for budgetary considerations"... can be coded as separate row entries such as:
      
      "The Board should conduct a vote every 3 years."
      
      "The Board should review proposal for budgetary considerations."

### Preprocessing/Anaphora Resolution (ABDICO_coreferences.ipynb)

Performs disambiguation of pronouns in policy documents or passages to resolve the absolute entity they are referring to.
This preserves valuable information by identifying the exact individuals, groups or organizational instruments associated with different activities and contexts.
Anaphora Resolution is recommended as a precursor preprocessing step to policy texts.

Input : .csv file where rows are passages/sections of policy documents (Best practice : language models have text limits. For best results, limit passages to 4 - 5 sentences. For even longer documents, break them down to such appropriate segments in each .csv row)

Output : All individual sentences from the policy documents/sections after their anaphora resolution

Example: After anaphora resolution, it becomes clear and specific that "them" in the policy refers to Podling websites

      Before:
            Statement: "there are restrictions on where podlings can host their websites and what branding they can use on them."
            Attribute : "Podlings" (observing restrictions)
            Objects : "their websites", "them"

      After Anaphora resolutions:
            Statement: "there are restrictions on where podlings can host their websites and what branding podlings can use on their websites"
            Attribute : "Podlings" (observing restrictions)
            Objects : "their websites", "their websites"

### Institutional Grammar Parsing (ABDICO_coreferences.ipynb)

