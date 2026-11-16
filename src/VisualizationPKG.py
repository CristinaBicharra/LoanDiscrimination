"""Plots used in the analysis. Every plot is saved as a PNG file."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_graph(df, x_variable, y_variable, graph_type, parameter, parameter_values,
               aux_bins, aux_x_label, aux_y_label, aux_title,
               output_dir='.', file_name=None, show=False):
    """Draw and save one plot.

    graph_type:
    - 'scatter': y against x, one color per group in `parameter_values`;
    - 'histogram': distribution of x, one color per group;
    - 'line': y against x, one line per group;
    - 'bins': number of clients of each group in bins of x (`aux_bins`);
    - 'solo_bins': bar chart of y for each category in x.

    Returns the path of the saved image.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if graph_type == 'scatter':
        for value in parameter_values:
            mask = df[parameter] == value
            ax.scatter(df.loc[mask, x_variable], df.loc[mask, y_variable],
                       alpha=0.6, s=8, label=value)
    elif graph_type == 'histogram':
        for value in parameter_values:
            sns.histplot(df.loc[df[parameter] == value, x_variable], kde=True,
                         label=value, bins=50, alpha=0.6, ax=ax)
    elif graph_type == 'line':
        for value in parameter_values:
            part = df[df[parameter] == value].sort_values(by=x_variable)
            ax.plot(part[x_variable], part[y_variable], label=value)
    elif graph_type == 'bins':
        binned = pd.cut(df[x_variable], bins=aux_bins, include_lowest=True)
        summary = df.groupby([binned, parameter], observed=False).size().unstack(fill_value=0)
        summary.plot(kind='bar', stacked=False, ax=ax)
    elif graph_type == 'solo_bins':
        ax.bar(df[x_variable].astype(str), df[y_variable], color='maroon', width=0.4)
    else:
        plt.close(fig)
        raise ValueError("Unsupported graph type. Choose from 'scatter', "
                         "'histogram', 'line', 'bins' or 'solo_bins'.")

    ax.set_xlabel(aux_x_label)
    ax.set_ylabel(aux_y_label)
    ax.set_title(aux_title)
    ax.tick_params(axis='x', rotation=45)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    if file_name is None:
        file_name = f"{x_variable}_{y_variable}_{graph_type}_{parameter}.png"
    path = os.path.join(output_dir, file_name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return path
