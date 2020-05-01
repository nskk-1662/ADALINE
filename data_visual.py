import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ADA_datagenerateor import df,X,y,X_std
from ADLINE_classs import AdalineGD

print(df)
#データの可視化
#グラフのプロット
plt.scatter(X[:50,0],X[:50,1],color = 'red',label = 'setosa')
plt.scatter(X[50:100,0],X[50:100,1],color = 'blue',label='versicolor')
plt.scatter(X[100:150,0],X[100:150,1],color='yellow',label='なんだっけ')
#軸ラベルの設定
plt.xlabel('speal length [cm]')
plt.ylabel('petal length [cm]')


#グラフのプロット(標準化データ)
plt.scatter(X_std[:50,0],X_std[:50,1],color = 'red',label = 'setosa')
plt.scatter(X_std[50:100,0],X_std[50:100,1],color = 'blue',label='versicolor')
plt.scatter(X_std[100:150,0],X_std[100:150,1],color='yellow',label='なんだっけ')

plt.show()
