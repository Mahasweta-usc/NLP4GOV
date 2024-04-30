!wget https://storage.googleapis.com/routines_semantic/IG_data/RedditRules.txt

!pip install jsonlines
import jsonlines
import json

subs = {}
with jsonlines.open('RedditRules.txt') as reader:
    for i, obj in enumerate(reader):
      if not i : print(obj.keys())
      rules = [elem['description'] for elem in obj['rules'] if elem['description'].strip()]
      sub = obj['sub']
      subs[sub] = {}

      subs[sub]['popularity'] = int(obj['subscribers'].replace(",",""))

      try: subs[sub]['rules'] = sub[sub].append(rules)
      except:
        subs[sub]['rules'] = rules

out_file = open("reddit_rules_full.json", "w")
json.dump(subs, out_file, indent = 4)

def keyfunc(tup):
    key, d = tup
    return d["popularity"]

subs = {k:v for k,v in sorted(subs.items(), key = keyfunc, reverse= True)[:100]}

print(subs)
# lens = [len(subs[k]['rules']) for k,v in subs.items()]
lens = [subs[k]['popularity'] for k,v in subs.items()]

from matplotlib import pyplot as plt
plt.hist(lens)


out_file = open("reddit_rules_top_100.json", "w")
json.dump(subs, out_file, indent = 4, separators = (', ',': '))

subs_list = list(subs.keys())