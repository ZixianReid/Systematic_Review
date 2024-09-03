import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import textwrap
import math
import matplotlib as mpl
from .tools import color_6

embedding_method = ['Event Embedding', 'GraphSAGE', 'InferCode', 'LSTM', 'RNN', 'TBCNN', 'Tree-CNN',
                    'cod2vec', 'doc2vec', 'graph2vec', 'word2vec']
exclusive_methods = ['Call Graph', 'Not clear', 'Semantic Flow Graph']


def plot_code_representation_name(df):
    df = df.assign(Code_feature_name=df['Code_feature_name'].str.split(',')).explode('Code_feature_name')
    df = df[~df['Code_feature_name'].isin(embedding_method + exclusive_methods)]

    counts = df['Code_feature_name'].value_counts().sort_index()
    plt.figure(figsize=(10, 10))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Code Feature Names')
    plt.show()


def plot_code_representation_type(df):
    font_size = 20

    # Set global font size
    mpl.rcParams['font.size'] = font_size  # or plt.rc('font', size=font_size)

    df = df.assign(data_representations_type=df['data_representations_type'].str.split(',')).explode(
        'data_representations_type')
    df = df[~df['data_representations_type'].isin(['Embedding-based', 'Not clear'])]



    counts = df['data_representations_type'].value_counts().sort_index()
    plt.figure(figsize=(12, 10))
    # plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=color_6, labeldistance=None)

    counts.plot(
        kind='pie',
        autopct='%1.1f%%',
        colors=color_6,
        labels=counts.index,  # Set the labels here
        labeldistance=None  # Suppresses labels on the chart
    )
    # plt.title('Distribution of Data Representation Types')
    plt.axis('equal')

    # Create legend with labels
    plt.legend(
        loc='upper center',  # Adjust this as needed to place the legend
        bbox_to_anchor=(0.5, 0.04),  # coordinates for legend placement
        fontsize=font_size,
        ncol=3  # This sets the number of columns in the legend
    )

    plt.ylabel('')  # Remove the y-axis label
    plt.subplots_adjust(left=0.0, right=0.85)  # Adjust as needed
    plt.tight_layout()
    plt.show()


def plot_code_representations_over_years(df):
    sns.set_theme(style="white")
    df['Year'] = df['Year'].apply(lambda x: str(int(x)))
    # df['Year'] = df['Year'].apply(lambda x: '2014 or earlier' if int(x) <= 2014 else x)
    df = df.assign(data_representations_type=df['data_representations_type'].str.split(',')).explode(
        'data_representations_type')
    df = df[~df['data_representations_type'].isin(['Embedding-based', 'Not clear'])]

    plt.figure(figsize=(12, 12))
    colors = color_6
    sns.histplot(data=df, x='Year', bins=45, edgecolor='black', hue='data_representations_type', palette=colors,
                 multiple='stack')

    types = df['data_representations_type'].unique()
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, types)]

    # Changed line
    legend_title = ""
    ncols_legend = math.ceil(len(types) / 2)

    plt.legend(handles=legend_patches, title=legend_title, loc='upper center', bbox_to_anchor=(0.5, 1.10),
               ncol=ncols_legend, fontsize=19)

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    plt.ylabel('Number of Primary Study', fontsize=24)
    plt.xlabel('', fontsize=24)
    plt.xticks(rotation=45)

    plt.tick_params(axis='both', which='major', labelsize=22)

    plt.tight_layout()  # Add this line to reduce blank area

    plt.show()


def table_code_representations(df):
    df = df[~df['data_representations_type'].isin(['Embedding-based', 'Not clear'])]
    df = df.assign(Code_feature_name=df['Code_feature_name'].str.split(',')).explode('Code_feature_name')
    df = df[~df['Code_feature_name'].isin(embedding_method + exclusive_methods)]

    with open('./source/citations_order.txt', 'r') as f:
        citations_order = [x.strip() for x in f.readline().split(',')]
    def print_cites(group):
        cites = group['cite'].tolist()
        output_list = []
        for cite in cites:
            item_name = cite.split('{')[1].split(',')[0]
            output_list.append(item_name)
        output_list.sort(key=lambda x: citations_order.index(x))
        output_str = ', '.join(output_list)
        print(f'{group.name}:{len(output_list)}')
        print(f'{group.name}: {output_str}')
    df.groupby('Code_feature_name').apply(print_cites)


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
    category_order = ['A' + str(i) for i in range(1, 16)]
    df['Application_type'] = pd.Categorical(df['Application_type'], categories=category_order, ordered=True)

    df = df.assign(data_representations_type=df['data_representations_type'].str.split(',')).explode(
        'data_representations_type')
    df = df[~df['data_representations_type'].isin(['Embedding-based', 'Not clear'])]

    colors = color_6

    # setting the figure size
    plt.figure(figsize=(12, 12))

    # Plotting
    sns.histplot(data=df, x='Application_type', bins=45, edgecolor='black', hue='data_representations_type', palette=colors,
                 multiple='stack')

    # Create Legend with correct labels and title
    legend_title = ""
    types = df['data_representations_type'].unique()
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, types)]
    plt.legend(handles=legend_patches, title=legend_title, loc='upper center', bbox_to_anchor=(0.50, 1.10),
               ncol=3, title_fontsize=14, fontsize=19)

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    plt.ylabel('Number of Primary Studies', fontsize=24)
    plt.xlabel('', fontsize=24)

    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.xticks(rotation=45)
    plt.tight_layout()  # Add this line to reduce blank area

    plt.show()

