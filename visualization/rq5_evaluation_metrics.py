import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


def table_evaluation_metrics(df):
    df = df[df['Evaluation_metrics'] != 'Not clear']
    df = df.assign(Evaluation_metrics=df['Evaluation_metrics'].str.split(',')).explode('Evaluation_metrics')

    print(df['Evaluation_metrics'].value_counts().sort_index())
    with open('./source/citations_order.txt', 'r') as f:
        citations_order = [x.strip() for x in f.readline().split(',')]

    def print_cites(group):
        cites = group['cite'].tolist()
        output_list = []
        for cite in cites:
            item_name = cite.split('{')[1].split(',')[0]
            output_list.append(item_name)
        # sort the output_list according to their order in citations_order,
        # push items not found in citations_order to the end of the list
        output_list.sort(key = lambda x: citations_order.index(x))
        output_str = ', '.join(output_list)
        print(f'{group.name}: {output_str}')

    df.groupby('Evaluation_metrics').apply(print_cites)

