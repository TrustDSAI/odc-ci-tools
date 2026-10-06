import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_bar(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Creates a Bar Graph
    
    Args:
        df (pd.DataFrame): DataFrame with the frequency of each element
        ax (plt.Axes): Graph's position in the subplot
    """
    
    # Convert the dataframe from wide to long format
    df_long = df.reset_index().melt(id_vars=df.index.name, var_name="Model", value_name="Frequency")
    bars = sns.barplot(data=df_long, x=df.index.name, y="Frequency", hue="Model", ax=ax, palette="tab20")
    
    # Only label bars that correspond to real rows in df_long
    frequencies = df_long["Frequency"].tolist()
    i = 0
    # Show bar labels (the number above the bar) for each defect
    for bar in bars.patches:
        # When all the values ends, stops adding bar labels, thus correcting the phantom 0s bug
        if i >= len(frequencies):
            break
        
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, int(frequencies[i]),
                ha='center', va='bottom', fontsize=6.5)
        i += 1
    
    # Labels
    ax.grid(axis="y", linestyle='--', alpha=0.4, linewidth=1)         # Adds a y-grid for better visualization
    ax.set_title(f"{df.index.name} by IA Model")
    ax.legend()

def create_pie(df: pd.DataFrame) -> None:
    """Creates multiple Pie Charts using matplotlib to show the proportion between the responses of each IA.
    
    Args:
        df (pd.DataFrame): DataFrame with the frequency of each IA response
    """
    
    # Only writes percent if it is greater than zero
    def func(pct):
        return f'{pct:.1f}%' if pct > 0 else '' 
    
    n = len(df.columns)
    n_cols = math.ceil(math.sqrt(n))     # Rounds the square root up
    n_rows = math.ceil(n / n_cols)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), num="Pie Chart - " + df.index.name, sharey=True)
    axs: np.ndarray[plt.Axes] = axs.flatten()       # Transforms the 2D matrix in a 1D matrix
    
    labels = df.index.tolist()
    cmap = plt.get_cmap("tab20")    # Palette with up to 20 different colors
    colors_map = {label: cmap(i / max(1, len(labels)-1)) for i, label in enumerate(labels)}
    
    for index, model in enumerate(df.columns):
        frequency = df.loc[:, model]
        frequency = frequency[frequency > 0]
        
        axs[index].pie(frequency, labels=None, autopct=func, colors=[colors_map[i] for i in frequency.index])
        axs[index].set_title(model)
    
    for j in range(n, len(axs)):
        fig.delaxes(axs[j])
    
    fig.suptitle(df.index.name)
    legend_handles = [plt.Line2D([0], [0], color=colors_map[label], lw=4) for label in labels]
    fig.legend(legend_handles, labels, title=df.index.name, loc="lower right")