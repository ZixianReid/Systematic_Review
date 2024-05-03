import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from .tools import color_3
import matplotlib as mpl

from sklearn.preprocessing import MinMaxScaler
import numpy as np

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def plot_clccd_performance(df):
    df = df[df['performance_atcoder'].notnull()]

    df['performance_atcoder'] = df['performance_atcoder'].apply(lambda x: [float(i) for i in eval(x)])

    df[['precision', 'recall', 'f1_score']] = pd.DataFrame(df['performance_atcoder'].tolist(), index=df.index)
    df.rename(columns={'precision': 'Precision', 'recall': 'Recall', 'f1_score': 'F1_score'}, inplace=True)

    # Manually specify the order of models
    model_order = ['CLCDSA', 'Perez et al', 'Ling et al', 'CICD-I', 'RUBHUS']
    df['model_name'] = df['model_name'].astype('category')
    df['model_name'] = df['model_name'].cat.reorder_categories(model_order)

    # Sort dataframe by 'model_name'
    df.sort_values("model_name", inplace=True)

    # Set 'model_name' as the dataframe index
    df.set_index('model_name', inplace=True)

    plt.figure(figsize=(12, 8))  # Increase figure size
    ax = df[['Precision', 'Recall', 'F1_score']].plot(kind='bar', color=color_3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=df.shape[1], fontsize=12)
    ax.set_xlabel('')  # Set x-label to an empty string

    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.17)  # Adjusted bottom margin
    plt.tight_layout()  # Add this line to reduce blank area

    plt.show()


def plot_ccd_performance_bcb(df):
    df = df[df['performance_bcb'].notnull()]
    df = df[df['model_name'].notnull()]
    df['performance_bcb'] = df['performance_bcb'].apply(lambda x: [float(i) for i in eval(x)])
    df[['Precision', 'Recall', 'F1_score']] = pd.DataFrame(df['performance_bcb'].tolist(), index=df.index)
    df.rename(columns={'precision': 'Precision', 'recall': 'Recall', 'f1_score': 'F1_score'}, inplace=True)

    model_order = ['CDLH', 'ASTNN', 'DeepSim', 'FCCA', 'FA-AST', 'TBCCD', 'SCCD-GAN', 'HELoC', 'PCAN',
                   'EA-HOLMES', 'Goner', 'Amain', 'TreeGen']
    df = df[df['model_name'].isin(model_order)]
    df['model_name'] = df['model_name'].astype('category')
    df['model_name'] = df['model_name'].cat.reorder_categories(model_order)

    df.sort_values("model_name", inplace=True)
    df.reset_index(inplace=True)

    plt.figure(figsize=(10, 12))
    ax = df[['Precision', 'Recall', 'F1_score']].plot(kind='bar', color=color_3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=df.shape[1], fontsize=12)

    # Custom x-axis
    plt.xticks(ticks=df.index, labels=df['model_name'].values, rotation=45)
    # Custom y-axis
    plt.ylim(0.6, 1.0)

    plt.subplots_adjust(bottom=0.17)
    # plt.grid(True)
    plt.tight_layout()  # Add this line to reduce blank area

    plt.show()


def plot_ccd_performance_ojclone(df):
    df = df[df['performance_ojclone'].notnull()]
    df = df[df['model_name'].notnull()]
    df['performance_ojclone'] = df['performance_ojclone'].apply(lambda x: [float(i) for i in eval(x)])
    df[['precision', 'recall', 'f1_score']] = pd.DataFrame(df['performance_ojclone'].tolist(), index=df.index)

    model_order = ['CDLH', 'ASTNN', 'TBCCD', 'MTN', 'HELoC', 'PCAN', 'TAILOR']
    df.rename(columns={'precision': 'Precision', 'recall': 'Recall', 'f1_score': 'F1_score'}, inplace=True)

    df = df[df['model_name'].isin(model_order)]
    df['model_name'] = df['model_name'].astype('category')
    df['model_name'] = df['model_name'].cat.reorder_categories(model_order)

    # Sort dataframe by 'model_name'
    df.sort_values("model_name", inplace=True)

    # Set 'model_name' as the dataframe index
    df.reset_index(inplace=True)

    plt.figure(figsize=(10, 12))

    ax = df[['Precision', 'Recall', 'F1_score']].plot(kind='bar', color=color_3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=df.shape[1], fontsize=12)
    # Custom x-axis
    plt.xticks(ticks=df.index, labels=df['model_name'].values, rotation=45)
    plt.ylim(0.4, 1.0)
    plt.subplots_adjust(bottom=0.17)  # Adjusted bottom margin
    plt.grid(False)
    plt.tight_layout()  # Add this line to reduce blank area

    plt.show()


def plot_ccd_performance_gcj(df):
    df = df[df['performance_gcj'].notnull()]
    df = df[df['model_name'].notnull()]
    df['performance_gcj'] = df['performance_gcj'].apply(lambda x: [float(i) for i in eval(x)])

    df[['precision', 'recall', 'f1_score']] = pd.DataFrame(df['performance_gcj'].tolist(), index=df.index)
    df.rename(columns={'precision': 'Precision', 'recall': 'Recall', 'f1_score': 'F1_score'}, inplace=True)
    model_order = ['DeepSim', 'FA-AST', 'SCCD-GAN', 'HELoC', 'EA-HOLMES', 'Goner', 'Amain', 'TreeGen']

    df = df[df['model_name'].isin(model_order)]
    df['model_name'] = df['model_name'].astype('category')
    df['model_name'] = df['model_name'].cat.reorder_categories(model_order)

    # Sort dataframe by 'model_name'
    df.sort_values("model_name", inplace=True)

    # Set 'model_name' as the dataframe index
    df.reset_index(inplace=True)

    plt.figure(figsize=(10, 12))

    ax = df[['Precision', 'Recall', 'F1_score']].plot(kind='bar', color=color_3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=df.shape[1], fontsize=12)

    # Custom x-axis
    plt.xticks(ticks=df.index, labels=df['model_name'].values, rotation=45)
    plt.ylim(0.6, 1.0)
    plt.subplots_adjust(bottom=0.17)  # Adjusted bottom margin
    plt.grid(False)
    plt.tight_layout()  # Add this line to reduce blank area

    plt.show()
