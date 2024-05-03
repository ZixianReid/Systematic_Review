import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from .tools import color_8
import matplotlib as mpl


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def table_open_source_dataset(df):
    df = df.assign(Dataset_name=df['Dataset_name'].str.split(',')).explode('Dataset_name')
    print(df['Dataset_name'].value_counts().sort_index())


def pie_chart_programming_languages(df):
    # Define your font size
    font_size = 20

    # Set global font size
    mpl.rcParams['font.size'] = font_size  # or plt.rc('font', size=font_size)

    df = df[df['Language_type'] != 'Not clear']
    df = df.assign(Language_type=df['Language_type'].str.split(',')).explode('Language_type')
    language_counts = df['Language_type'].value_counts().sort_index()

    # plot pie chart
    plt.figure(figsize=(10, 12))
    language_counts.plot.pie(autopct="%1.1f%%", colors=color_8, labeldistance=None)
    plt.legend(
        loc='upper center',  # Adjust this as needed to place the legend
        bbox_to_anchor=(0.5, 0.08),  # coordinates for legend placement
        fontsize=font_size,
        ncol=4  # This sets the number of columns in the legend
    )
    plt.axis('equal')
    plt.ylabel('')  # This removes 'Language_type' ylabel because it is already in the title
    # plt.title('Distribution of Language Types')
    # Adjust the subplot size so that the legend fits into the figure
    plt.subplots_adjust(left=0.0, right=0.85)  # Adjust as needed
    plt.tight_layout()
    plt.show()


def datasets_over_applications(df):
    with open('./source/citations_order.txt', 'r') as f:
        citations_order = [x.strip() for x in f.readline().split(',')]

    df = df[df['Dataset_name'] != 'Not clear']
    df = df.assign(Dataset_name=df['Dataset_name'].str.split(',')).explode('Dataset_name')

    grouped = df.groupby(['Application_type', 'Dataset_name']).size().reset_index(name='Count')

    def sort_cites(group):
        cites = group['cite'].tolist()
        output_list = []
        for cite in cites:
            item_name = cite.split('{')[1].split(',')[0]
            output_list.append(item_name)
        # sort the output_list according to their order in citations_order,
        # push items not found in citations_order to the end of the list
        output_list.sort(key=lambda x: citations_order.index(x))
        return ", ".join(output_list)

    grouped['cites'] = df.groupby(['Application_type', 'Dataset_name']).apply(sort_cites).values

    print(grouped)
