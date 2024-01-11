import pandas as pd

processed = pd.DataFrame(columns=["raw institutional statement", "Attribute", "Object", "Deontic", "Aim"])
for no in range(1, 19):
    data = pd.read_csv(f'fpc_{no}_with_code.csv')
    index = data.index[data['word'] == 'ROOT'].tolist()

    for idx, elem in enumerate(index[:-1]):
        sub = data[index[idx]:index[idx + 1]]
        sub = sub[(sub['word'] != 'ROOT') & (sub['relation'] != 'punct')]
        sub.sort_values(by=['tid'])
        statement = " ".join(sub['word'].to_list())

        att = " ".join(sub[(sub['CodeType'] == 'Attribute')]['word'].to_list())
        obj = " ".join(sub[(sub['CodeType'] == 'Object')]['word'].to_list())
        deon = " ".join(sub[(sub['CodeType'] == 'Deontic')]['word'].to_list())
        aim = " ".join(sub[(sub['CodeType'] == 'Aim')]['word'].to_list())
        entry = pd.DataFrame([[statement, att, obj, deon, aim]],
                             columns=["raw institutional statement", "Attribute", "Object", "Deontic", "Aim"])
        print(entry.head())
        processed = pd.concat([processed, entry])

print(processed.shape[0])
processed.to_csv("sample_fpc.csv", index=False)