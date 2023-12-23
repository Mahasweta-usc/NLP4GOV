import json
import pandas as pd
# Opening JSON file
f = open('/content/IG-SRL/policy_comparison/data/reddit_rules_top_100.json')

# returns JSON object as
# a dictionary
data = json.load(f)

community1 = data[drop_down.value]['rules']
db1 = pd.DataFrame(community1, columns=["Raw Institutional Statement"])
community2 = data[dependent_drop_down.value]['rules']
db2 = pd.DataFrame(community2, columns=["Raw Institutional Statement"])

db1.to_csv("/content/db1.csv",index=False)
db2.to_csv("/content/db2.csv",index=False)