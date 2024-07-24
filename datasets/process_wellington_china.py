from sklearn import preprocessing
import numpy as np
import pandas as pd

import urllib.request

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

# China's gross domestic product - Produto interno bruto chinês
url = 'https://raw.githubusercontent.com/Blackman9t/Machine_Learning/master/china_gdp.csv'

filename = "china_gdp_wellington.csv"

urllib.request.urlretrieve(url, filename)

data = np.genfromtxt(filename,  skip_header=1, delimiter=",")

np.random.seed(0)
np.random.shuffle(data)
fronteira = round(data.shape[0] * 0.8)
scaler = preprocessing.MinMaxScaler()
data = scaler.fit_transform(data)

train = data[:fronteira]
test = data[fronteira:]
np.savetxt("China-Train.csv", train, delimiter=" ",
           header="x0 response", fmt="%.6f")
np.savetxt("China-Test.csv", test, delimiter=" ",
           header="x0 response", fmt="%.6f")
# Normalizando para melhorar os resultados
# scaler = preprocessing.MinMaxScaler()
# df['Year'] = scaler.fit_transform(df[['Year']])
# df['Value'] = scaler.fit_transform(df[['Value']])
