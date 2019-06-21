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
TimeLSTMは、T個分の時系列データをまとめて処理するレイヤ。

RNNで学習を行う際は、Truncated BPTTを行う。
- 逆伝播のつながりを適当な長さで断ち切る。
- 順伝播の流れは維持する。
  - 隠れ状態と記憶セルをメンバ変数に保持させる。

```python
    class TimeLSTM:
        def __init__(self, Wx, Wh, b, stateful=False):
            self.params = [Wx, Wh, b]
            self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
            self.layers = None

            self.h, self.c = None, None
            self.dh = None
            self.stateful = stateful

        def forward(self, xs):
            Wx, Wh, b = self.params
            N, T, D = xs.shape
            H = Wh.shape[0]

            self.layers = []
            hs = np.empty((N, T, H), dtype='f')

            if not self.stateful or self.h is None:
                self.h = np.zeros((N, H), dtype='f')
            if not self.stateful or self.c is None:
                self.c = np.zeros((N, H), dtype='f')

            for t in range(T):
                layer = LSTM(*self.params)
                self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
                hs[:, t, :] = self.h

                self.layers.append(layer)

            return hs

        def backward(self, dhs):
            Wx, Wh, b = self.params
            N, T, H = dhs.shape
            D = Wx.shape[0]

            dxs = np.empty((N, T, D), dtype='f')
            dh, dc = 0, 0

            grads = [0, 0, 0]
            for t in reversed(range(T)):
                layer = self.layers[t]
                dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
                dxs[:, t, :] = dx
                for i, grad in enumerate(layer.grads):
                    grads[i] += grad

            for i, grad in enumerate(grads):
                self.grads[i][...] = grad
            self.dh = dh
            return dxs

        def set_state(self, h, c=None):
            self.h, self.c = h, c

        def reset_state(self):
            self.h, self.c = None, None
```


## 6.4　LSTMを使った言語モデル
- 5章で実装した「RNNを使った言語モデル」とほとんど同じ。
- Time RNNレイヤを Time LSTMレイヤに変える。

Rnnlmクラスの実装

```python
    class Rnnlm(BaseModel):
        def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
            V, D, H = vocab_size, wordvec_size, hidden_size
            rn = np.random.randn

            # 重みの初期化
            embed_W = (rn(V, D) / 100).astype('f')
            lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
            lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
            lstm_b = np.zeros(4 * H).astype('f')
            affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
            affine_b = np.zeros(V).astype('f')

            # レイヤの生成
            self.layers = [
                TimeEmbedding(embed_W),
                TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
                TimeAffine(affine_W, affine_b)
            ]
            self.loss_layer = TimeSoftmaxWithLoss()
            self.lstm_layer = self.layers[1]

            # すべての重みと勾配をリストにまとめる
            self.params, self.grads = [], []
            for layer in self.layers:
                self.params += layer.params
                self.grads += layer.grads

        def predict(self, xs):
            for layer in self.layers:
                xs = layer.forward(xs)
            return xs

        def forward(self, xs, ts):
            score = self.predict(xs)
            loss = self.loss_layer.forward(score, ts)
            return loss

        def backward(self, dout=1):
            dout = self.loss_layer.backward(dout)
            for layer in reversed(self.layers):
                dout = layer.backward(dout)
            return dout

        def reset_state(self):
            self.lstm_layer.reset_state()
```

学習のためのコード

```python
    import sys
    sys.path.append('..')
    from common.optimizer import SGD
    from common.trainer import RnnlmTrainer
    from common.util import eval_perplexity
    from dataset import ptb
    from rnnlm import Rnnlm


    # ハイパーパラメータの設定
    batch_size = 20
    wordvec_size = 100
    hidden_size = 100  # RNNの隠れ状態ベクトルの要素数
    time_size = 35  # RNNを展開するサイズ
    lr = 20.0
    max_epoch = 4
    max_grad = 0.25

    # 学習データの読み込み
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_test, _, _ = ptb.load_data('test')
    vocab_size = len(word_to_id)
    xs = corpus[:-1]
    ts = corpus[1:]

    # モデルの生成
    model = Rnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)

    # 勾配クリッピングを適用して学習
    trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad,
                eval_interval=20)
    trainer.plot(ylim=(0, 500))

    # テストデータで評価
    model.reset_state()
    ppl_test = eval_perplexity(model, corpus_test)
    print('test perplexity: ', ppl_test)

    # パラメータの保存
    model.save_params()
```

## 6.5　RNNLMのさらなる改善
上記で説明したRNNLMの改善ポイント３点

- LSTMレイヤの多層化
- Dropout
- 重み共有

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

