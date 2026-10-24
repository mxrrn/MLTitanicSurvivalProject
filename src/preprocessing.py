'''
Optimized preprocessing
'''

import pandas as pd
from age_modelling import model_ages

def preprocess_data(file_path, data_info=''):
    '''
    Preprocess the data

    Parameters:
    file_path (str): The path to the CSV file containing the data.
    data_info (str): Additional information to include in the printed output (default is '').

    Returns:
    pd.DataFrame: Preprocessed feature data.
    pd.Series: Target variable.
    '''
    # Load data
    data = pd.read_csv(file_path)

    # Preprocessing the data
    # Display first rows and data structure
    print(f"First rows of the {data_info} dataset:")
    print(data.head())
    print(f"\nData structure of the {data_info} dataset:")
    print(data.info())

    # Analyze missing values
    print(f"\nMissing values per column in the {data_info} dataset:")
    print(data.isnull().sum())

    # Convert categorical data to numerical
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})  # Convert 'Sex'
    
    # Fill missing values for 'Fare'
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Fare'] = data['Fare'].astype(int)

    # Fill missing values for 'Age' using the model_ages function
    data = model_ages(data)

    # Fill missing values for 'Embarked'
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # Handle missing values for 'Cabin' as 'Unknown' and extract only the first letter (Deck)
    data['Cabin'] = data['Cabin'].fillna('Unknown')
    data['Cabin'] = pd.Categorical(data['Cabin']).codes

    # Create dummy variables for 'Embarked'
    data = pd.get_dummies(data, columns=['Sex', 'Pclass', 'Embarked'], dtype=int)

    # Additional features
    data['TicketFrequency'] = data['Ticket'].map(data['Ticket'].value_counts())
    data['IsAloneTicket'] = (data['TicketFrequency'] == 1).astype(int)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
    data['Family_Size_Grouped'] = data['FamilySize'].map(family_map)
    data = pd.get_dummies(data, columns=['Family_Size_Grouped'], dtype=int)

    # Select features
    data.drop(['Name','Ticket','PassengerId','FamilySize'], axis=1, inplace=True)

    # Prepare input data (X) and target variable (y)
    X = data.drop('Survived', axis=1)  # All columns except 'Survived'
    y = data['Survived']  # Target variable

    # Display prepared data
    print(f"\nPrepared input data (X) for the {data_info} dataset:")
    print(X.head())
    print(f"\nTarget variable (y) for the {data_info} dataset:")
    print(y.head())
    X.dropna(inplace=True)
    y = y[X.index]

    return X, y
