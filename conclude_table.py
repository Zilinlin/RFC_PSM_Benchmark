'''
This file is to conclude the results of a json file to severla tables
'''


import json
import pandas as pd

with open("fsm_evaluation_results.json", "r") as infile:
    data = json.load(infile)

# Flatten the JSON into a DataFrame
records = []
for key, metrics_dict in data.items():
    protocol, model = key.split('_', 1)
    for partial_option, stats in metrics_dict.items():
        records.append({
            'Protocol': protocol,
            'Model': model,
            'Partial': partial_option,
            'Precision': stats['precision'],
            'Recall': stats['recall'],
            'F1-score': stats['f1_score']
        })

df = pd.DataFrame(records)

# Generate pivot tables for with_partial and no_partial
# 'with_partial', 
for partial_option in ['with_partial']:
    df_subset = df[df['Partial'] == partial_option].pivot_table(
        index='Protocol',
        columns='Model',
        values=['F1-score']
    )
    print(f"% LaTeX table for {partial_option}\n")
    print(df_subset.to_latex(float_format="%.3f", multirow=True))
    print("\n")