import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


class AdalineGD:

    def __init__(self,eta = 0.01,n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter

    def init_weight(self, X):
        self.w_ = np.random.normal(0, 0.01, size=X.shape[1] + 1)

    def fit(self,X,y):
        self.init_weight(X)
        self.cost_ = [] # 性能評価のため取っておく

        #トレーニング回数分トレーニングデータを反復
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            #誤差の計算
            errors = (y-output)
            #w1以降の更新
            self.w_[1:] += self.eta*X.T.dot(errors)
            #w0の更新
            self.w_[0] += self.eta*errors.sum()
            #コスト関数の計算(勾配降下法)
            cost = (errors**2).sum()/2.0
            #コストの格納
            self.cost_.append(cost)
        return self

    #挿入力の計算
    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]

    #活性化関数の出力を計算
    def activation(self,X):
        return X

    #1ステップ後のクラスラベルを返す
    def predict(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.0,1,-1)


def plot_decision_regions(X,y,classifier,resolution = 0.02):
    #マーカーとカラーマップの用意
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #決定領域のプロット
    x1_min,x1_max = X[:,0].min() - 1,X[:,0].max() + 1
    x2_min,x2_max = X[:,1].min() - 1,X[:,1].max() + 1
    #グリッドポイントの作成
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                         np.arange(x2_min,x2_max,resolution))
    #特徴量を一次元配列に変換して予測を実行する
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    #予測結果をグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    #グリッドポイントの等高線のプロット
    plt.contourf(xx1,xx2,Z,alpha = 0.4,cmap = cmap)
    #軸の範囲の設定
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    #クラスごとにサンプルをプロット
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y == cl,1],alpha=0.8,c = cmap(idx),marker=markers[idx],label=cl)
