import pandas as pd
import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

#1-150番目まで目的変数の抽出
y = df.iloc[:,4].values
#Iris-setosa を-1　iris-verginica を1に変換
y = np.where(y == 'Iris-setosa',-1,1)
#1-150行目の1,3列めの抽出
X = df.iloc[:,[1,3]].values
print(X)
#データの標準化s
X_std = np.copy(X)
#各列の標準化
X_std[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()
