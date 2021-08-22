import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# carregando o arquivo
baseData = pd.read_csv('svbr.csv', delimiter=';')
#print(baseData)

# carregando os dados sem o nome da coluna
X = baseData.iloc[:, :].values
#print(X)

# define o tipo de valor que sera subistituido e a strategia para o calculo
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# faz o calculo
imputer = imputer.fit(X[:, 1:3])
# transforma o valor em X em string
X = imputer.transform(X[:, 1:3]).astype(str)

# insere em X a coluna 0
X = np.insert(X, 0, baseData.iloc[:, 0].values, axis=1)
print(X)
