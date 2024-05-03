import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from .tools import color_3


def chart_ml_type_over_years(df):
    sns.set_theme(style="whitegrid")

    df['Year'] = df['Year'].apply(lambda x: str(int(x)))
    df['Year'] = df['Year'].apply(lambda x: '2014 or earlier' if int(x) <= 2014 else x)
    df = df.assign(Machine_learning_type=df['Machine_learning_type'].str.split(',')).explode('Machine_learning_type')

    plt.figure(figsize=(10, 12))

    # Match the colors to the provided figure
    colors = color_3  # Assuming the order matches the provided figure
    sns.histplot(data=df, x='Year', bins=45, edgecolor='black', hue='Machine_learning_type', palette=colors,
                 multiple='stack')

    # Create Legend with correct labels and title
    legend_title = ""
    types = df['Machine_learning_type'].unique()
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, types)]
    plt.legend(handles=legend_patches, title=legend_title, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=len(types), title_fontsize=14, fontsize=14)

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    plt.ylabel('Number of Primary Study', fontsize=16)
    plt.xlabel('Year', fontsize=16)  # Set the fontsize of x-label
    plt.show()


def table_ml(df):
    df = df.assign(machine_learning_name=df['machine_learning_name'].str.split(',')).explode('machine_learning_name')
    # counts = df['machine_learning_name'].value_counts().sort_index()
    # print(counts)
    deep_learning_list = ["BiLSTM", "CNN", 'CapsNet', 'DNN', 'GAT', 'GCN', 'GGNN', 'GMN', 'GNN', 'GRU', 'LSTM', 'RAE',
                          'RNN', 'RSGNN', 'Siamese network', 'TCN', 'Transformer', 'Tree-CNN', 'Tree-LSTM',
                          'Code2vec']
    deep_learning_list = sorted(deep_learning_list)
    conventional_machine_learning = ['ANN', 'Adaboost', 'Bagging', 'Decision Table', 'Decision Tree', 'GDBT', 'IBK',
                                     'J48',
                                     'KNN', 'Logistic regression', 'Logit Boost',
                                     'Naive Bayes', 'Random Committee', 'Random Forest', 'Random Subspace',
                                     'Random Tree',
                                     'Rotation Forest', 'SVM', 'Stochastic Grad.D. Classifier', 'XGboost']
    conventional_machine_learning = sorted(conventional_machine_learning)
    clustering_learning = ['DBSCAN', 'K-means', ]
    clustering_learning = sorted(clustering_learning)
    # print(deep_learning_list)
    # print(conventional_machine_learning)
    # print(clustering_learning)

    df = df[~df['machine_learning_name'].isin(['GPT-2', 'Not clear'])]
    # counts = df['machine_learning_name'].value_counts().sort_index()
    def print_cites(group):
        cites = group['cite'].tolist()
        output_list = []
        for cite in cites:
            item_name = cite.split('{')[1].split(',')[0]
            output_list.append(item_name)
        output_str = ', '.join(output_list)
        print(f'{group.name}:{len(output_list)}')
        print(f'{group.name}: {output_str}')
    df.groupby('machine_learning_name').apply(print_cites)


def plot_ml_over_applications(df):
    def function(x):
        if x in ['code clone', 'code clone,source code classification']:
            return 'Code clone'
        elif x in ['Cross-Language Code Clone,software similarity', 'software similarity']:
            return 'software similarity'
        else:
            return x

    dict_app_code = {'Code clone': 'A1', 'Cross-Language Code Clone': 'A2', 'code clone validation': 'A3',
                     'application clone': 'A4', 'software similarity': 'A5',
                     'code similarity': 'A6', 'Plagiarism detection': 'A7',
                     'Software plagiarism detection': 'A8', 'Vulnerability detection': 'A9',
                     'code change inspection': 'A10', 'code prediction': 'A11',
                     'software review': 'A12', 'program repair': 'A13', 'Issue-Commit Link Recovery': 'A14',
                     'algorithm classification': 'A15'}

    df['Application_type'] = df['Application_type'].apply(function)
    df['Application_type'] = df['Application_type'].map(dict_app_code)

    # Convert 'Application_type' to categorical type and specify order
    category_order = ['A' + str(i) for i in range(1, 16)]
    df['Application_type'] = pd.Categorical(df['Application_type'], categories=category_order, ordered=True)

    df = df.assign(Machine_learning_type=df['Machine_learning_type'].str.split(',')).explode('Machine_learning_type')
    colors = color_3

    sns.histplot(data=df, x='Application_type', bins=45, edgecolor='black', hue='Machine_learning_type', palette=colors,
                 multiple='stack')

    # Create Legend with correct labels and title
    legend_title = ""
    types = df['Machine_learning_type'].unique()
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, types)]
    plt.legend(handles=legend_patches, title=legend_title, loc='upper center', bbox_to_anchor=(0.50, 1.09),
               ncol=len(types), title_fontsize=14, fontsize=14)

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    plt.ylabel('Number of Primary Study')
    plt.show()
    # sns.countplot(data=df, x='Application_type', hue='Machine_learning_type')
    # plt.xlabel('Application Type')
    # plt.ylabel('Count')
    # plt.xticks(rotation=45)
    # plt.show()


if __name__ == '__main__':
    pass
    # df = pd.DataFrame({
    #     'id': [1, 2],
    #     'machine_learning_name': ['a,b', 'c,d,e']
    # })
    #
    # df = df.assign(machine_learning_name=df['machine_learning_name'].str.split(',')).explode('machine_learning_name')
