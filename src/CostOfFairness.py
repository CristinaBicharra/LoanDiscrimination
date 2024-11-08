#!/usr/bin/env python
# coding: utf-8

# In[1]:


#calculate_fairness_cost:
#Purpose: Adjusts interest rates with a fixed discount across a specific credit score range, aiming to equalize rates for a discriminated group up to a certain category.
#Core Calculation: Total cost is the sum of adjustments for each category up to max_category, based on loan amount, interest rate difference, fairness degree, and non-default probability.

#calculate_fairness_cost_with_decay:
#Purpose: Adjusts interest rates with a decaying discount for each lower credit score category, providing a gradual decrease in fairness adjustments as credit score worsens.
#Core Calculation: Total cost is calculated similarly but incorporates a decay_factor, making the discount smaller with each lower category, ensuring a minimum adjustment through min_discount.

import numpy as np

def calculate_fairness_cost(df, Loan_var, CreditScoreBin, parameter, Discriminator, Discriminated, price_discriminator, fairness_degree, max_category, Default_var):
    """
    Calculates the total cost for the bank to mitigate discrimination by adjusting interest rates to approximate
    the rates offered to a dominant group. The function applies a fixed discount up to a specified credit score
    category, aligning the interest rate of the discriminated group to that of the dominant group.

    Parameters:
    - df (pd.DataFrame): DataFrame containing loan information and demographic data.
    - Loan_var (str): Column name for loan amount.
    - CreditScoreBin (str): Column representing credit score categories (bins).
    - parameter (str): Column representing the demographic group (e.g., 'Gender').
    - Discriminator (str): Dominant group in the demographic parameter (e.g., 'Male').
    - Discriminated (str): Group that experiences discrimination (e.g., 'Female').
    - price_discriminator (str): Column representing the interest rate or similar price variable.
    - fairness_degree (float): Degree to which the interest rate should be adjusted towards fairness.
    - max_category (int): Maximum number of credit score categories to consider in the adjustment.
    - Default_var (str): Column representing whether the client defaulted (1 if defaulted, 0 otherwise).

    Returns:
    - float: Rounded total cost of implementing fairer rates for the discriminated group.

    Steps:
    1. Sorts the data by credit score category (`CreditScoreBin`).
    2. Computes the average interest rates for both groups (Discriminator and Discriminated) in each category.
    3. For each category up to `max_category`, calculates the cost if the rate for the discriminated group
       exceeds that of the dominant group.
    4. The cost is based on the loan amount, difference in interest rates, number of contracts, non-default rate,
       and fairness degree.
    """

    data = df.copy()
    data = data.sort_values(by=CreditScoreBin, ascending=True)
    labels = data[CreditScoreBin].unique()

    avg_interest = data.groupby([CreditScoreBin, parameter], observed=False)[price_discriminator].mean().unstack()

    total_cost = 0

    for i in range(max_category):
        bin_label = labels[i]
        if Discriminated in avg_interest.columns and Discriminator in avg_interest.columns:
            discrim_rate = avg_interest.loc[bin_label, Discriminated]
            dom_rate = avg_interest.loc[bin_label, Discriminator]

            if not np.isnan(discrim_rate) and not np.isnan(dom_rate) and discrim_rate > dom_rate:
                difference = discrim_rate - dom_rate
                reduced_difference = difference

                bin_data = data[(data[CreditScoreBin] == bin_label) & (data[parameter] == Discriminated)]
                avg_loan_amount = bin_data[Loan_var].mean()
                num_contracts = len(bin_data)
                avg_non_default = 1 - bin_data[Default_var].mean()

                cost = avg_loan_amount * reduced_difference * fairness_degree * num_contracts * avg_non_default
                total_cost += cost

    return round(total_cost, 2)


def calculate_fairness_cost_with_decay(data, Loan_var, CreditScoreBin, parameter, price_discriminator, discriminated, discriminator, max_discount, min_discount, max_category, decay_factor):
    """
    Calculates the cost for the bank to mitigate discrimination with a discount that decays as the credit score
    decreases. The discount reduces the interest rate for the discriminated group, gradually decreasing with
    each lower credit score category.

    Parameters:
    - data (pd.DataFrame): DataFrame containing loan information and demographic data.
    - Loan_var (str): Column name for loan amount.
    - CreditScoreBin (str): Column representing credit score categories (bins).
    - parameter (str): Column representing the demographic group (e.g., 'Gender').
    - price_discriminator (str): Column representing the interest rate or similar price variable.
    - discriminated (str): Group that experiences discrimination (e.g., 'Female').
    - discriminator (str): Dominant group in the demographic parameter (e.g., 'Male').
    - max_discount (float): Maximum discount to be applied to the interest rate difference.
    - min_discount (float): Minimum discount to ensure a baseline adjustment even with decay.
    - max_category (int): Maximum number of credit score categories to consider in the adjustment.
    - decay_factor (float): Factor by which the discount decays as credit score decreases.

    Returns:
    - float: Total cost of applying a decaying discount to the interest rate for the discriminated group.

    Steps:
    1. Sorts data by credit score category (`CreditScoreBin`).
    2. Calculates the average interest rates for both groups (discriminator and discriminated) in each category.
    3. For each category up to `max_category`, calculates the cost if the rate for the discriminated group
       exceeds that of the dominant group.
    4. Applies a decaying discount to the difference in interest rates, reducing as credit score worsens.
    5. The cost is based on the loan amount, reduced difference with discount, and number of contracts.

    """

    avg_interest = data.groupby([CreditScoreBin, parameter], observed=True)[price_discriminator].mean().unstack()
    total_cost = 0

    labels = data[CreditScoreBin].unique()

    for i in range(max_category):
        bin_label = labels[i]
        if discriminated in avg_interest.columns and discriminator in avg_interest.columns:
            discrim_rate = avg_interest.loc[bin_label, discriminated]
            dom_rate = avg_interest.loc[bin_label, discriminator]

            if not np.isnan(discrim_rate) and not np.isnan(dom_rate) and discrim_rate > dom_rate:
                difference = discrim_rate - dom_rate

                discount = max(max_discount - (decay_factor * i), min_discount)

                reduced_difference = difference * discount

                bin_data = data[(data[CreditScoreBin] == bin_label) & (data[parameter] == discriminated)]
                avg_loan_amount = bin_data[Loan_var].mean()
                num_contracts = len(bin_data)

                cost = avg_loan_amount * reduced_difference * num_contracts
                total_cost += cost

    return round(total_cost, 2)


# In[ ]:




