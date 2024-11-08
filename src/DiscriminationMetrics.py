#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Different methoids for evaluating fairness
# calculate_weighted_disparity_index enphasizes demographic parity
# equality_of_opportunity_M1 and equality_of_opportunity_M2 calculates equality of opportunity
# calculate_equality_of_opportunity calculates equality of opportunity and equity odds

import numpy as np

def calculate_weighted_disparity_index(bin_summary, discriminated, discriminator):
    import numpy as np
# Calculate Weighted Disparity Index (WDI)
# - The Weighted Disparity Index (WDI) provides an overall measure of whether males are
# - disproportionately receiving lower interest rates compared to females,
# - with higher weights given to the bins with more favorable (lower) interest rates.
#
# WDI varies from -1 to 1.
# 1 is the highest discrimination,
# 0 is the lowest and
# -1 means the discrimination might be the other way around

    # Calculate Weighted Disparity Index (WDI)
    weights = np.exp(-np.arange(len(bin_summary)))  # Higher weights for lower interest rate bins
    weighted_disparities = []

    for bin_index in range(len(bin_summary)):
        discriminator_count = bin_summary.iloc[bin_index][discriminator]
        discriminated_count = bin_summary.iloc[bin_index][discriminated]
        bin_count=discriminator_count+discriminated_count
        if bin_count > 0:
            disparity_ratio = (discriminator_count-discriminated_count)/bin_count
        else:
            disparity_ratio = 0  # Avoid division by zero
        weighted_disparity = disparity_ratio * weights[bin_index]
        #print(disparity_ratio)
        weighted_disparities.append(weighted_disparity)

    # Calculate overall Weighted Disparity Index
    wdi = np.sum(weighted_disparities) / np.sum(weights)
    return wdi


def equality_of_opportunity_M1(df_aux, parameter, discriminated, discriminator,
                                      price_discriminator, credit_score_bins):
    

    df=df_aux.copy()
    df=df.sort_values(by=credit_score_bins,ascending=True)
    TNC=len(df) # total number of clients

    opportunity_ratios = []
    eop_bins=[]
    bins=[]
    # Iterate over each credit score bin
    for credit_bin in df[credit_score_bins].unique():
        df_bin = df[df[credit_score_bins] == credit_bin] # clients in credit bin
        print('Oi',credit_bin, df[df[credit_score_bins] == credit_bin].CreditScore.mean())
        print('fonte de erro')
        print(df_bin)
        print('fim')

        NC=len(df_bin) # number of clients in the bin
        NC_discriminated= len(df_bin[df_bin[parameter]==discriminator] )
        NC_discriminator=len(df_bin[df_bin[parameter]==discriminated])
        print('NC',NC, NC_discriminator,NC_discriminated)
        print('-->',df_bin[df_bin[parameter]==discriminator] )
        print('<**>',df_bin[df_bin[parameter]==discriminated] )
        #print(df_bin[df_bin[parameter]==discriminator][price_discriminator])
        
        if NC_discriminator==0:
            rate_disparity_range=0
            rate_disparity=0
        else:
            if NC_discriminated==0:
                average_rate_discriminated=0
            else:
                # Calculate the proportion of the target gender receiving favorable interest rates
                average_rate_discriminated=round(np.nanmean(df_bin[df_bin[parameter]==discriminated][price_discriminator]),3)
            #print(average_rate_discriminator, average_rate_discriminated)
            
            average_rate_discriminator=round(np.nanmean(df_bin[df_bin[parameter]==discriminator][price_discriminator]),3)

            #Interest Rate Disparity (IRD)
            rate_disparity=average_rate_discriminated - average_rate_discriminator # price difference

            rate_disparity_range=rate_disparity*NC # price difference multiplied by the size of the cluster-bin

        opportunity_ratios.append(rate_disparity_range)
        eop_bin = round(rate_disparity,2)
        eop_bins.append(eop_bin)
        bins.append(credit_bin)

    # Calculate the equality of opportunity
    over_all_eop = round(np.nansum(opportunity_ratios) /TNC,2)
    print('dentro')
    print(eop_bins)
    print('---')
    print(bins)
    print('saindo')

    return round(over_all_eop,2), eop_bins, bins



# Function to evaluate fairness criteria in the dataset using credit score bins and display results in a formatted table
def calculate_equality_of_odds_opportunity(df_aux,
                                                       parameter,
                                                       discriminated,
                                                       discriminator,
                                                       price_discriminator,
                                                       cs_bin_var,

                                                       par_default,
                                                       default_threshold):
    """
    Evaluates fairness in terms of Equality of Opportunity, Equality of Odds
    using credit score bins and displays the results in a formatted table similar to the provided example.
    It returns thre values: the euqlaity of opportunity index, equality of odds and a table with the value of inequalities
    for non-default (good to calculate eq. of opportunity) and default values (good to calculate eq. of odds together with non-defaults) credit score bin

    Parameters:
        df: The dataset to be evaluated.
        parameter: the name of the attribute of suspected discrimination, such as Gender
        discriminated: the value of the discriminated group
        discriminator: the value of the dominant group
        price_discriminator: the attribute where the discrimination ocurs, such as Interest Rate
        cs_bin_var: the attribute name of the credit score categories (bins)
        cs_bins_labels: the possible values of cs_bin_var, i.e. the categories values
        par_default: the name of the attribute in which to record default probailities
        default_threshold: the threshold to consider for par_default to decide if the client will default or not
    """
    df=df_aux.copy()
    df=df.sort_values(by=cs_bin_var,ascending=True)
    # Initialize list to store results
    table_data = []
    opportunity_ratios=[]
    eop_bins=[]
    eoo_bins=[]
    odds_ratio=[]
    bins=[]

    AC=len(df) # all clients
    ACND=len(df[df[par_default]<=default_threshold]) # number of clinets that did not default or has a low probalility of default
    ACD=len(df[df[par_default]>default_threshold])# number of clients that did default or has a high probalility of default
    labels=df[cs_bin_var].unique()

    # Loop through each category and calculate fairness metrics
    for label in labels:
        bins.append(label)
        # Equality of Opportunity: Non-Defaults
        non_defaulters = df[(df[cs_bin_var] == label) & (df[par_default] <= default_threshold )]
        NCND=len(non_defaulters)
        if NCND>0:
            avg_interest_rate_discriminator_non_def = non_defaulters[non_defaulters[parameter] == discriminator][price_discriminator].mean()
            avg_interest_rate_discriminated_non_def = non_defaulters[non_defaulters[parameter] == discriminated][price_discriminator].mean()

            #Interest Rate Disparity (IRD)
            rate_disparity=avg_interest_rate_discriminated_non_def - avg_interest_rate_discriminator_non_def # price difference

            rate_disparity_range=rate_disparity*NCND/AC # price difference multiplied by the size of the cluster-bin
        else:
            rate_disparity = 0
            rate_disparity_range=0

        opportunity_ratios.append(rate_disparity_range)
        eop_bin = round(rate_disparity,2)
        eop_bins.append(eop_bin)


        # Equality of Odds: Defaults
        defaulters = df[(df[cs_bin_var] == label) & (df[par_default] > default_threshold )]
        NCDefault=len(defaulters)
        if NCDefault>0:
            avg_interest_rate_discriminator_def = defaulters[defaulters[parameter] == discriminator][price_discriminator].mean()
            avg_interest_rate_discriminated_def = defaulters[defaulters[parameter] == discriminated][price_discriminator].mean()

            #Interest Rate Disparity Defaulters(IRDD)
            rate_disparity_def=avg_interest_rate_discriminated_def - avg_interest_rate_discriminator_def # price difference
            rate_disparity_def_range=rate_disparity_def*NCDefault/AC # price difference multiplied by the size of the cluster-bin
        else:
            rate_disparity_def=0
            rate_disparity_def_range=0

        eoo_bin = round(rate_disparity_def,2)
        eoo_bins.append(eoo_bin)

        odds_ratio.append(rate_disparity_def_range)

    # Calculate the equality of opportunity
    over_all_eop = round(np.nansum(opportunity_ratios) ,2)
    over_all_eoo= round((np.nansum(odds_ratio) + np.nansum(opportunity_ratios)),2)
    data = {
        'Credit Score Bins': bins,
        'Non-Default': opportunity_ratios,
        'Default': odds_ratio
    }

    df_equality = pd.DataFrame(data)

    return over_all_eop,  over_all_eoo, df_equality




def calculate_equality_of_opportunity(df_aux, parameter, discriminated, discriminator,
                                      price_discriminator, credit_score_bins):
    
    '''
Equality of Opportunity Metric

1- Divide the data into credit score bins and interest rate bins.

2- Calculating Average Interest Rate by Group (Female/Male) and Credit Score Range
   
3- Interest Rate Disparity (IRD): Calculate the disparity (difference) in interest rates between female and male clients

4. Weight by the number of clients in each credit score bin

5- Calculate the Overall Discrimination Index
To get an overall discrimination measure across all credit scores, you can take a weighted average of
IRD values across all credit score ranges, weighted by the number of clients in each range

## Interpretation:

Overall IRD=0: No discrimination, as similar credit scores yield similar interest rates across genders.

Overall IRD>0: On average, women are charged higher interest rates than men for similar credit scores, indicating possible price discrimination.

Overall IRD<0: On average, men are charged higher interest rates than women.
'''
    df=df_aux.copy()
    df=df.sort_values(by=credit_score_bins,ascending=True)
    TNC=len(df) # total number of clients

    opportunity_ratios = []
    eop_bins=[]
    bins=[]
    # Iterate over each credit score bin
    for credit_bin in df[credit_score_bins].unique():
        df_bin = df[df[credit_score_bins] == credit_bin] # clients in credit bin
        #print('Oi',credit_bin, df[df[credit_score_bins] == credit_bin].CreditScore.mean())

        NC=len(df_bin) # number of clients in the bin

        # Calculate the proportion of the target gender receiving favorable interest rates
        average_rate_discriminator=round(np.nanmean(df_bin[df_bin[parameter]==discriminator][price_discriminator]),3)
        average_rate_discriminated=round(np.nanmean(df_bin[df_bin[parameter]==discriminated][price_discriminator]),3)
        #print(average_rate_discriminator, average_rate_discriminated)

        #Interest Rate Disparity (IRD)
        rate_disparity=average_rate_discriminated - average_rate_discriminator # price difference

        rate_disparity_range=rate_disparity*NC # price difference multiplied by the size of the cluster-bin

        opportunity_ratios.append(rate_disparity_range)
        eop_bin = round(rate_disparity,2)
        eop_bins.append(eop_bin)
        bins.append(credit_bin)

    # Calculate the equality of opportunity
    over_all_eop = round(np.nansum(opportunity_ratios) /TNC,2)

    return round(over_all_eop,2), eop_bins, bins


# In[ ]:





# In[ ]:




