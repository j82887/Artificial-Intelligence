## 實作 - 貓狗影像資料集
https://www.kaggle.com/c/dogs-vs-cats/data

## 01. 安裝套件
1. pip install keras==1.0.8
2. pip install tensorflow== 1.13.1 --user 
* 本教材只裝於Keras版本為 1.08，Tensorflow版本為 1.13.1。若要嘗試最新Tensorflow 2.0版本可以參考網址安裝：https://tf.wiki/zh_hant/basic/installation.html
* 安裝不成功者則可以使用Colab進行實作

* **安裝過程中可出現下列BUG：** 
1. 在CMD中：`WARNING: The scripts xx.exe are installed in 'C:\Users\...' which is not on PATH. Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.`，解決方式是將該位置加入環境變數中，從「本機」中點擊右鍵選擇「內容」，它會跳到"控制台\系統及安全性\系統"，選擇「進階系統設定」→「環境變數」，將該環境添加進去
2. 在import tensorflow 部分套件中：`FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'._np_quint8 = np.dtype([("quint8", np.uint8, 1)])`，原因是numpy太新，解決方式降numpy版本 `pip install numpy==1.16.0 --user`
3. 在import keras 部分套件中：`ImportError: cannot import name 'to_categorical'`，原因可能為keras版本部分與tensorflow不符，因此從pip改使用conda來安裝`conda install keras`或是從`from keras.utils import to_categorical`改用`from keras.utils.np_utils import to_categorical`
