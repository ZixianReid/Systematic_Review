import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import math
import matplotlib as mpl
from .tools import color_1, color_6, color_5


def plot_year(df):
    # Define your font size
    font_size = 20

    # Set global font size
    mpl.rcParams['font.size'] = font_size

    print(df['Application_type'].unique())

    df['Year'] = df['Year'].apply(lambda x: str(int(x)))
    # df['Year'] = df['Year'].apply(lambda x: '2014 or earlier' if int(x) <= 2014 else x)
    year_counts = df['Year'].value_counts().sort_index()
    colors = color_1

    plt.figure(figsize=(10, 12))
    bars = plt.bar(year_counts.index, year_counts, color=colors)
    plt.ylabel('Number of Primary Study', fontsize=font_size)
    plt.xticks(rotation=45, va='top', fontsize=font_size)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', fontsize=font_size)

    # plt.xlabel('Years', fontsize=font_size, labelpad=-15)
    plt.tight_layout()  # Add this line to reduce blank area
    plt.show()


def plot_applications_per_year(df):

    def function(x):
        if x in ['code clone', 'code clone validation', 'code clone label',
                 'code clone,source code classification', 'code clone validation', 'application clone']:
            return 'Code clone'
        elif x in ['Cross-Language Code Clone,software similarity', 'code similarity', 'software similarity']:
            return 'Code similarity'
        elif x in ['Software plagiarism detection', 'Plagiarism detection']:
            return 'Plagiarism detection'
        elif x == 'Vulnerability detection':
            return 'Vulnerability detection'
        elif x == 'code change inspection':
            return 'Others'
        elif x == 'code prediction':
            return 'Others'
        elif x == 'software review':
            return 'Others'
        elif x == 'program repair':
            return 'Others'
        elif x == 'Issue-Commit Link Recovery':
            return 'Others'
        elif x == 'algorithm classification':
            return 'Others'
        elif x == 'Cross-Language Code Clone':
            return 'Cross-Language Code Clone'

    sns.set_theme(style="whitegrid")
    df['Year'] = df['Year'].apply(lambda x: str(int(x)))
    # df['Year'] = df['Year'].apply(lambda x: '2014 or earlier' if int(x) <= 2014 else x)
    df['Application_type'] = df['Application_type'].apply(function)

    plt.figure(figsize=(10, 12))
    colors = color_6
    sns.histplot(data=df, x='Year', bins=45, edgecolor='black', hue='Application_type', palette=colors,
                 multiple='stack')

    legend_title = ""
    types = df['Application_type'].unique()
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, types)]

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))

    plt.xlabel('', fontsize=20)
    plt.ylabel('Number of Primary Study', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)

    ncols_legend = math.ceil(len(types) / 2)
    plt.legend(handles=legend_patches, title=legend_title, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=ncols_legend, fontsize=15)
    plt.tight_layout()  # Add this line to reduce blank area
    plt.show()

def plot_application(df):
    # Define your font size
    font_size = 20

    # Set global font size
    mpl.rcParams['font.size'] = font_size  # or plt.rc('font', size=font_size)

    def function(x):
        if x in ['Cross-Language Code Clone', 'code clone', 'code clone validation', 'code clone label',
                'code clone,source code classification', 'code clone validation', 'application clone']:
            return 'Code clone'
        elif x in ['Cross-Language Code Clone,software similarity', 'code similarity', 'software similarity']:
            return 'Code similarity'
        elif x in ['Software plagiarism detection', 'Plagiarism detection']:
            return 'Plagiarism detection'
        elif x == 'Vulnerability detection':
            return 'Vulnerability detection'
        elif x == 'code clone,code change inspection':
            return 'Others'
        elif x == 'code clone,code prediction':
            return 'Others'
        elif x == 'software review':
            return 'Others'
        elif x == 'program repair':
            return 'Others'
        elif x == 'Issue-Commit Link Recovery':
            return 'Others'
        elif x == 'algorithm classification':
            return 'Others'

    df['Application_type'] = df['Application_type'].apply(function)

    # Define colors
    colors = color_5  # Ensure this variable contains a list of color codes

    # Plotting the pie chart
    fig, ax = plt.subplots(figsize=(10, 12))
    counts = df['Application_type'].value_counts()
    pie_labels = counts.index.tolist()  # Labels for the slices
    pie = counts.plot(
        kind='pie',
        autopct='%1.1f%%',
        colors=colors,
        labels=pie_labels,  # Set the labels here
        labeldistance=None  # Suppresses labels on the chart
    )
    plt.axis('equal')

    # Create legend with labels
    plt.legend(
        loc='upper center',  # Adjust this as needed to place the legend
        bbox_to_anchor=(0.5, 0.04),  # coordinates for legend placement
        fontsize=font_size,
        ncol=2  # This sets the number of columns in the legend
    )

    plt.ylabel('')  # Remove the y-axis label

    # Adjust the subplot size so that the legend fits into the figure
    plt.subplots_adjust(left=0.0, right=0.85)  # Adjust as needed
    plt.tight_layout()
    plt.show()

def count_application_number(df):
    def function(x):
        if x in ['code clone', 'code clone,source code classification']:
            return 'Code clone'
        elif x in ['Cross-Language Code Clone,software similarity', 'software similarity']:
            return 'software similarity'
        else:
            return x

    df['Application_type'] = df['Application_type'].apply(function)
    print(df['Application_type'].value_counts())