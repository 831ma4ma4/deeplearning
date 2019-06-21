6章　ゲート付きRNN（後半）
====

<ゴール>  
LSTMの実装を行う。言語モデルを作り、実際のデータでうまく学習できることを確認する。


## 6.3　LSTMの実装

- 1ステップを処理するクラスを**LSTMクラス**として実装する。
- Tステップ分をまとめて処理するクラスを**TimeLSTMクラス**として実装する。

LSTMクラスで行う計算は以下の通り。  
![](https://github.com/831ma4ma4/deeplearning/blob/master/6-3-01.PNG)

記憶セルの計算  
隠れ状態の計算  

上記４つのアフィン変換（xW<sub>x</sub>+hW<sub>h</sub>+b）は、ひとつの式でまとめて計算することが出来る。  

LSTMクラスの初期化  

```python
    class LSTM:
        def __init__(self, Wx, Wh, b):
            '''
            Parameters
            ----------
            Wx: 入力`x`用の重みパラーメタ（4つ分の重みをまとめる）
            Wh: 隠れ状態`h`用の重みパラメータ（4つ分の重みをまとめる）
            b: バイアス（4つ分のバイアスをまとめる）
            '''
            self.params = [Wx, Wh, b]
            self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
            self.cache = None
```

順伝播の実装
```python
    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next
```

### 6.3.1　TimeLSTMの実装
aa

## 6.4　LSTMを使った言語モデル
いいい

## 6.5　RNNLMのさらなる改善
あああい

### 6.5.1　LSTMレイヤの多層化
あああ

### 6.5.2　Dropoutによる過学習の抑制
あああ
![](https://github.com/831ma4ma4/deeplearning/blob/master/6-5-2-01.PNG)  
![](https://github.com/831ma4ma4/deeplearning/blob/master/6-5-2-02.PNG)  

- 層を深くすることでモデルの表現力が増し、複雑な依存関係を学習することが期待できるが、過学習を起こしやすくなる。  
  - 過学習とは、訓練データだけに対して正しい答えを出し、汎化能力が欠如した状態を指す。
    - aaaa

- 変分ドロップアウト（Variational Dropout）  
同じ階層にあるドロップアウトでは、共通のマスクを利用する。（マスクは「データを通す/通さない」の二値のランダムパターン）


### 6.5.3　重み共有
あああ

### 6.5.4　より良いRNNLMの実装
あああ

### 6.5.5　最先端の研究へ

PTBデータセットに対する各モデルのパープレキシティの結果
![](https://github.com/831ma4ma4/deeplearning/blob/master/6-5-5-01.PNG)  
https://arxiv.org/abs/1708.02182


## まとめ

言語モデルの改善テクニック
- LSTMレイヤの多層化
- Dropout
- 重み共有

