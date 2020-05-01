import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ADA_datagenerateor import df,X,y,X_std
from ADLINE_classs import AdalineGD,plot_decision_regions

#描画領域を1行２列に変換
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,4))

# #勾配降下法によるAdaLineGDの学習
# ada1 = AdalineGD(n_iter=100,eta=0.01).fit(X,y)
# #エポック数とコストの関係を示す折れ線グラフのプロット
# ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
# #軸ラベルの設定
# ax[0].set_xlabel=('Epochs')
# ax[0].set_ylabel=('sum-squared-error')
# #タイトルの設定
# ax[0].set_title('Adaline - Learning rate 0.01')
# #勾配降下法によるAdaLineGDの学習
# ada2 = AdalineGD(n_iter=100,eta=0.0001).fit(X,y)
# #エポック数とコストの関係を示す折れ線グラフのプロット
# ax[1].plot(range(1,len(ada2.cost_)+1),ada2.cost_,marker='o')
# #軸ラベルの設定
# ax[1].set_xlabel=('Epochs')
# ax[1].set_ylabel=('sum-squared-error')
# #タイトルの設定
# ax[1].set_title('Adaline - Learning rate 0.0001')

#勾配降下法によるADALINEの学習(標準化後)
ada = AdalineGD(n_iter=150,eta=0.01)
#モデルの適合
ada.fit(X_std,y)
#境界領域のプロット
plot_decision_regions(X_std,y,classifier=ada)
#タイトルの設定
plt.title('Adaline - Gradient Descent')
#軸ラベルの設定
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
#凡例の設定
plt.legend(loc = 'upper left')
# 保存先のファイル名(PNG形式)
fname = "gradient.png"
plt.savefig(fname, dpi = 128, tight_layout = True)
plt.show()
