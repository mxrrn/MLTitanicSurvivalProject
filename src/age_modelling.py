'''
Age imputation
'''

import pandas as pd


def model_ages(data):
    '''
    Fill missing values in the 'Age' column.

    Each missing age is replaced by the median age of passengers with the same ticket class and sex.
    Groups without any known age fall back to the overall median age.

    Parameters:
    data (pd.DataFrame): Passenger data with the columns 'Age', 'Pclass' and 'Sex' ('male' / 'female').

    Returns:
    pd.DataFrame: The data with all ages filled.
    '''
    data = data.copy()
    group_median = data.groupby(['Pclass', 'Sex'])['Age'].transform('median')
    data['Age'] = data['Age'].fillna(group_median).fillna(data['Age'].median())
    return data
