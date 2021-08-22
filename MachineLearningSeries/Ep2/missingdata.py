import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# carregando o arquivo
baseData = pd.read_csv('admission.csv', delimiter=';')
# print(baseData)

# carregando os dados sem o nome da coluna
X = baseData.iloc[:, :-1].values
y = baseData.iloc[:, -1].values

# define o tipo de valor que sera subistituido e a strategia para o calculo
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# faz o calculo
imputer = imputer.fit_transform(X[:, 1:])

# substitui o nome das pessoas por id
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])

# substitui o id por binary
X = X[:, 1:]
D = pd.get_dummies(X[:, 0])
X = np.insert(X, 0, D.values, axis=1)
print(X)

# separando o conjunto de teste e traino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(X_test)