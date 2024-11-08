#!/usr/bin/env python
# coding: utf-8

# In[2]:


# src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_graph(df, x_variable, y_variable, graph_type, parameter, parameter_values, aux_bins, aux_x_label, aux_y_label, aux_title):
    """
    Generates customized plots for analyzing interest rate discrimination among clients.

    The `plot_graph` function supports different types of plots (scatter, histogram, line, and binned),
    allowing visualization of how interest rates vary based on credit score and demographic group.
    These plots are useful for interpreting variations in interest rates based on demographic factors.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to be visualized.
    - x_variable (str): Column name for the X-axis.
    - y_variable (str): Column name for the Y-axis (used for scatter and line plots).
    - graph_type (str): Desired plot type. Options are:
        - 'scatter': Generates a scatter plot, useful to observe the relationship between interest rate and credit score.
        - 'histogram': Plots the distribution of values in the X-axis variable, categorized by the parameter.
        - 'line': Generates a line plot, suitable to observe trends over the X-axis.
        - 'bins': Creates a bar plot showing the distribution of clients in different interest rate bins.
    - parameter (str): Column used to categorize (e.g., 'gender').
    - parameter_values (list): List of parameter values to plot, e.g., [0, 1] for 'gender' (0 for male, 1 for female).
    - aux_bins (int or list): Number of bins (for 'histogram') or list of bin edges (for 'bins').
    - aux_x_label (str): X-axis label.
    - aux_y_label (str): Y-axis label.
    - aux_title (str): Title of the plot.

    Plot and Metric Interpretation:
    - The scatter plot ('scatter') shows how interest rates vary with credit scores for different demographic groups.
    - The bar plot ('bins') displays the distribution of male and female clients across interest rate ranges.
    - The histogram ('histogram') allows us to visually assess if the interest rate follows a normal distribution among groups.

    Example usage:
        plot_graph(df, 'credit_score', 'interest_rate', 'scatter', 'gender', [0, 1], None, 'Credit Score', 'Interest Rate', 'Interest Rate by Credit Score and Gender')

    Returns:
    - None. The function displays the plot directly and saves an image with an auto-generated name.

    Errors:
    - ValueError: If `graph_type` is not one of the supported types ('scatter', 'histogram', 'line', or 'bins').

    """
    
    plt.figure(figsize=(10, 6))
    if graph_type == 'scatter':
        # Scatter plot to visualize relationship between x and y, categorized by 'parameter'
        for value in parameter_values:
            plt.scatter(df[x_variable][df[parameter] == value], df[y_variable][df[parameter] == value], alpha=0.6, label=value)
    
    elif graph_type == 'histogram':
        # Histogram to view the distribution of x values, categorized by 'parameter'
        for value in parameter_values:
            sns.histplot(df[x_variable][df[parameter] == value], kde=True, label=value, bins=50, alpha=0.6)
    
    elif graph_type == 'line':
        # Line plot to view the trend of y over x, categorized by 'parameter'
        for value in parameter_values:
            df_value = df[df[parameter] == value].sort_values(by=x_variable)
            plt.plot(df_value[x_variable], df_value[y_variable], label=value)
    
    elif graph_type == 'bins':
        # Divide the variable of interest into bins and count clients in each bin by category
        df['aux_bins'] = pd.cut(df[x_variable], bins=aux_bins, include_lowest=True)
        
        # Summary count of each category in each bin
        bin_summary = df.groupby(['aux_bins', parameter]).size().unstack(fill_value=0)
        
        # Bar plot to show distribution among bins
        bin_summary.plot(kind='bar', stacked=False, color=['green', 'maroon'], figsize=(10, 6))
    elif graph_type == 'solo_bins':
        print('entrei certo')
        plt.bar(df[x_variable], df[y_variable], color='maroon', width=0.4)
    
    else:
        raise ValueError("Unsupported graph type. Choose from 'scatter', 'histogram', 'line', or 'bins'.")

    plt.xlabel(aux_x_label)
    plt.ylabel(aux_y_label)
    plt.title(aux_title)
    plt.xticks(rotation=45)
    plt.legend()
    
    # Save plot as an image
    name_fig = f"{x_variable}_{y_variable}_{graph_type}_{parameter}.png"
    plt.savefig(name_fig, dpi=300, bbox_inches='tight')
    plt.show()


# In[ ]:




