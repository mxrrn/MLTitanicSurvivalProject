import pandas as pd

def preprocess_data(train_path='train.csv', test_path='test.csv'):
    # Daten laden
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Erste Zeilen und Datenstruktur anzeigen
    print("Erste Zeilen des Trainingsdatensatzes:")
    print(train_data.head())
    print("\nDatenstruktur:")
    print(train_data.info())

    # Fehlende Werte analysieren
    print("\nFehlende Werte pro Spalte:")
    print(train_data.isnull().sum())

    # 1. Fehlende Werte auffüllen
    # Fehlende Werte für 'Age' auffüllen
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())

    # # Fehlende Werte für 'Embarked' auffüllen
    # train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

    # 2. Fehlende Werte für 'Cabin' behandeln 'Unknown' und extrahieren nur den ersten Buchstaben (Deck)
    train_data['Cabin'] = train_data['Cabin'].fillna('Unknown')
    # train_data['Cabin'] = train_data['Cabin'].apply(lambda x: x[0])
    # train_data = pd.get_dummies(train_data, columns=['Cabin'], drop_first=True)
    # train_data = pd.get_dummies(train_data, columns=['Cabin'], prefix='Cabin', dtype=int)
    train_data['Cabin'] = pd.Categorical(train_data['Cabin']).codes

    # 3. Kategorische Daten in numerische umwandeln
    train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})  # 'Sex' umwandeln

    # 4. Dummy-Variablen für 'Embarked'
    train_data = pd.get_dummies(train_data, columns=['Embarked'], dtype=int)

    # 5. Unnötige Spalten entfernen
    train_data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

    # 6. Eingabedaten (X) und Zielvariable (y) vorbereiten
    X = train_data.drop('Survived', axis=1)  # Alle Spalten außer 'Survived'
    y = train_data['Survived']  # Zielvariable

    # 7. Ausgabe der vorbereiteten Daten
    print("\nVorbereitete Eingabedaten (X):")
    print(X.head())
    print("\nZielvariable (y):")
    print(y.head())

    return train_data

if __name__ == "__main__":
    train_data = preprocess_data()
    print(train_data.head())
