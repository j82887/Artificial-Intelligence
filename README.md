# Artificial-Intelligence

![GITHUB](https://www.edntaiwan.com/wp-content/uploads/sites/6/images/7447e8f3-110b-4a56-82e9-9cf540145331.gif)

## Class 01 人工智慧 Artificial-Intelligence [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/01_Artificial-Intelligence/01_%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7.pdf)
* 人工智慧發展史(The History of Artificial Intelligence)
* 人工智慧的任務(The Basic Objectives of Artificial Intelligence)
  * 回歸(Regression)
  * 分類(Classification)
  * 分群(Cluster)
  * 複雜性任務(Complicated Mission)
* 人工智慧的學習方法
  * 非監督式學習(Unsupervised Learning)
  * 監督式學習(Supervised Learning)
  * 強化學習(Reinforcement Learning)
* 資料格式
  * 結構化資料(Sturctured Data)
  * 半結構化資料(Semi-Structured Data)
  * 非結構化資料(Unstructured Data)
* 人工智慧相關的軟體工程師
* **實作 01-01 Google Colab**
* **實作 01-02 本機使用Anaconda安裝Python、建立環境與安裝套件**

---
## Class 02 機器學習 - 線性回歸 Linear-Regression [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/02_Linear-Regression/02_%E7%B7%9A%E6%80%A7%E5%9B%9E%E6%AD%B8.pdf)
* 參數與超參數(Parameters and Hyperparameters)
* 損失函數(Loss Function): 均方誤差 (Mean Squared Error,MSE)
* 梯度下降法(Gradient Descent, GD)
* 學習速率(Learning Rate)
* 批量梯度下降法(Batch Gradient Descent, BGD)
* 隨機梯度下降法(Stochastic Gradient Descent, SGD)
* 小批量梯度下降法(Mini-batch Gradient Descent, MBGD)
* 線性回歸實作
* **實作 02-01 身體質量指數預測**

---
## Class 03 機器學習 - 邏輯回歸 Logistic Regression [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/03_Logistic-Regression/03_%E9%82%8F%E8%BC%AF%E5%9B%9E%E6%AD%B8.pdf)
* 邏輯/乙狀函數(Logistic / Sigmoid Function)
* 最大似然函數(Maximum Likelihood Function)
* 勝算比(Odds Ratio)
* 正則化(Regularization)
* 邏輯回歸實作
* **實作 03-1 數字手寫辨識**
* **實作 03-2 糖尿病預測**

---
## Class 04 機器學習 - 決策樹與隨機森林 Decision Tree and Random Forest [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/04_Decision-Tree%26Random-Forest/04_%E6%B1%BA%E7%AD%96%E6%A8%B9%E8%88%87%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97.pdf)
* 決策樹架構
  * 分支標準：交叉熵與基尼不純度(Cross-Entropy and Gini Impurity)
  * 訊息增益量(Information Gain)
  * 剪枝(Pruning)
    * 預剪枝(Pre-pruning)
    * 後剪枝(Post-pruning)
* 隨機森林架構
  * 集成學習(Ensemble Learning)：裝袋(Bagging)、取樣(Sampling)
* 決策樹與隨機森林實作
* **實作 04 字母手寫辨識**

---
## Class 05 機器學習 - 自適應增強 AdaBoost [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/05_AdaBoost/05_%E8%87%AA%E9%81%A9%E6%87%89%E5%A2%9E%E5%BC%B7.pdf)
* 提升(Boosting)
* 自適應增強(Adaptive Boosting)
* 集成學習：重新加權訓練集 (Reweighting Training Data)
  * 均勻權重(Uniform Weight)
  * 非均勻權重(Non-uniform Weight)
* 弱分類器(Weak Classifier)
* 決策樹樁(Decision Stump)

---
## Class 06 深度學習 - 多層感知器 Multilayer Perceptron [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/06_Multilayer%20perceptron/06_%E5%A4%9A%E5%B1%A4%E6%84%9F%E7%9F%A5%E5%99%A8.pdf)
* 深度神經網路(Deep Neural Network)
  * 輸入層(Input Layer)
  * 隱藏層(Hidden Layer)
  * 輸出層(Output Layer)
* 前向傳播(Forward Propagation)
* 後向傳播(Backward Propagation)
* 線性可分與線性不可分(Linear and Non-linear Separability)
* 梯度消失(Vanishing Gradient)
* 激勵函數(Activation Function)
* 丟棄法(Dropout)
* 多層感知器實作
* **實作 06 貓狗影像辨識**

---
## Class 07-1 深度學習 - 卷積神經網路 Convolutional Neural Network [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/07_Convolutional-Neural-Network/07_%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF.pdf)
* 卷積層(Convolution Layer)
  * 濾波器/卷積核(Filter / Convolution Kernel)
  * 遮罩/窗口(Mask)
  * 卷積/旋積(Convolution)
  * 填充(Padding)
  * 特徵圖(Feature Map)
* 池化層(Pooling Layer)
  * 局部最大池化(Local Maximum Pooling)
  * 局部平均池化(Local Average Pooling)
  * 局部隨機池化(Local Random Pooling)
  * 全域最大池化(Global Maximum Pooling)
  * 全域平均池化(Global Average Pooling)
* 全連接層(Fully Connected Layer)
  * Softmax
* 自動提取特徵(Automatic Feature Extraction)
* 權重共享(Shared Weight)
* 卷積神經網路實作
* **實作 07 Cifar100 CNN辨識**

## Class 07-2 深度學習 - 卷積神經網路架構的演化史 The evolutionary history of CNN architectures [PDF]
* LeNet 1998 [[Paper]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
* AlexNet 2012 [[Paper]](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
* VGGNet 2014 [[Paper]](https://arxiv.org/pdf/1409.1556.pdf)
* Inception(GoogLeNet) 2014 [[Paper]](https://arxiv.org/pdf/1409.4842.pdf)
* ResNet 2015 [[Paper]](https://arxiv.org/pdf/1512.03385.pdf)
* Res2Net 2019 [[Paper]](https://arxiv.org/pdf/1904.01169.pdf)

---
## Class 08 深度學習 - 自編碼器 Auto-Encoder [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/08_Auto-Encoder/08_%E8%87%AA%E7%B7%A8%E7%A2%BC%E5%99%A8.pdf)
* 編碼器(Encoder)
* 解碼器(Decoder)
* 去噪自編碼器(Denoising Auto-Encoder)
* 降維(Dimension Reduction)
* 資料視覺化(Data Visualization)
* 權重初始化(Weight Initialization)

---
## Class 09 深度學習 - 生成對抗神經網路 Generative Adversarial Network [[PDF]](https://github.com/j82887/Artificial-Intelligence/raw/master/09_Generative-Adversarial-Network/09_%E7%94%9F%E6%88%90%E5%B0%8D%E6%8A%97%E7%B6%B2%E8%B7%AF.pdf)
* 生成器(Generator) 
* 鑑別器(Discriminator) 
* 結構化學習(Structured Learning)
  * Bottom-Up 
  * Top-Down
* Cycle GAN

---
## Class 10 分類模型的驗證指標 Classification Model Assessment and Validation [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/10_Classification-Model-Assessment-and-Validation/10_%E5%88%86%E9%A1%9E%E6%A8%A1%E5%9E%8B%E7%9A%84%E9%A9%97%E8%AD%89%E6%8C%87%E6%A8%99.pdf)
* 混淆矩陣(Confusion Matrix)
* 準確度(Accuracy)
* 特異度(Specificity)
* 敏感度/召回率(Sensitivity / Recall)
* 精確度(Precision)
* F1分數(F1 Score)
* ROC曲線(Receiver Operating Characteristic Curve, ROC Curve)
* PR曲線(Precision Recall Curve, PR Curve)
* 曲線下面積(Area Under Curve, AUC)
* 交叉驗證(Cross-validation, CV)
  * 分層交叉驗證(Stratified Cross-validation, SCV)
  * 留一法交叉驗證(Leave One Out Cross-validation, LOOCV)
  * 隨機分割交叉驗證(Random Cross-validation, RCV)
  * K格交叉驗證(K-fold Cross-validation)
* 學習曲線(Learning Curve)

---
## Class 11 資料前處理 - 不均勻類別問題 Class Imbalance Problems [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/11_Class-Imbalance-Problem/11_%E8%B3%87%E6%96%99%E5%89%8D%E8%99%95%E7%90%86%20-%20%E4%B8%8D%E5%9D%87%E5%8B%BB%E9%A1%9E%E5%88%A5%E5%95%8F%E9%A1%8C.pdf)
* 取樣(Sampling)
  * 欠取樣(Under-sampling)
  * 過取樣(Over-sampling)
* 合成少數類別過取樣技術(Synthetic Minority Over-sampling Technique)
* Tomek Link(T-Link)
* 編輯最近鄰演算法(Edited Nearest Neighbor, ENN)

---
## Class 12 資料前處理 - 特徵縮放 與 獨熱編碼 Feature Scaling and One Hot Encoding [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/12_Feature-Scaling_and_One-Hot-Encoding/12_%E8%B3%87%E6%96%99%E5%89%8D%E8%99%95%E7%90%86%20-%20%E7%89%B9%E5%BE%B5%E7%B8%AE%E6%94%BE%E8%88%87%E7%8D%A8%E7%86%B1%E7%B7%A8%E7%A2%BC.pdf)
* 特徵縮放(Feature Scaling)
  * 重新縮放(Rescaling / Min-max Normalization)
  * 平均值正規化(Mean Normalization)
* 特徵處理/編碼(Feature Processing / Coding)
  * 排序行編碼(Sequential Encoding)
  * 獨熱編碼(One Hot Encoding)
  
---  
## Class 13 超參數搜索 Hyperparameter Search [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/13_Hyperparameter-Search/13_%E8%B6%85%E5%8F%83%E6%95%B8%E6%90%9C%E7%B4%A2.pdf)
* 均格/網格搜索(Grid Search)
* 隨機搜索(Random Search)
* 拉丁超立方抽樣(Latin Hypercube Sampling, LHS)
* 貝葉斯優化(Bayesian Optimization)
* 基因演算法(Genetic AlgorithmsGA)
---
## Class 14 梯度下降法的優化 Optimization of Gradient Descent Method [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/14_Optimization%20of%20Gradient%20Descent%20Method/14_%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95.pdf)
* 梯度下降法(Gradient Descent)
* 動量的梯度下降法(Gradient Descent with Momentum)
* AdaGrad(Adaptive Gradient)
* 均方根傳播(Root Mean Square Propagation, RMSProp)
* Adam(Adaptive Moment Estimation)

---
## Class 15 分類器的機率切點 Probability Thresholds of Classifier [PDF]

---
## Class 16 資料後處理 Data Post-processing [PDF]
* 非最大抑制(Non-maximum suppression, NMS)
* 軟非最大抑制(Soft Non-maximum suppression, Soft-NMS)
* 較軟非最大抑制(Softer Non-maximum suppression, Softer-NMS)
* 距離並交比非最大抑制(Distance IoU Non-maximum suppression, DIoU-NMS)

---
## Class 17 目標檢測模型 Object Detection [[PDF]](https://github.com/j82887/Artificial-Intelligence/raw/master/17_Object%20Detection/17_%E7%9B%AE%E6%A8%99%E6%AA%A2%E6%B8%AC%E6%A8%A1%E5%9E%8B.pdf)
* 技術發展(Technological development)
* 基於哈爾特徵級聯分類器(Haar Feature-based Cascade Classifiers)
* 非最大抑制(Non-maximum suppression, NMS)
* 並交比(Intersection-over-UnionIoU, IoU)
* 廣義並交比(Generalized Intersection over Union, GIOU) **#尚未補充**
* 距離並交比(Distance-IoU, DIoU) **#尚未補充**
* 完整並交比(Complete-IoU, CIoU) **#尚未補充**
* 平均精度(Average Precision, AP)
* 平均精度均值(mean average precision, mAP)

---
## Class 18 其他學習方法 Other Learning Methods [PDF]
* 遷移學習(Transfer Learning, TL)
  * 微調(Fine-tune)
* 聯邦學習(Federated Learning, FL)
  * 橫向聯邦學習(Horizontal Federated Learning, HFL)
  * 縱向聯邦學習(Vertical Federated Learning, VFL)
  * 聯邦遷移學習(Federated Transfer Learning, FTL)
* 邊緣AI(Edge AI)

---
## Class 19  相關實作 Implementation [PDF]
* 車牌辨識
* 人臉辨識
* 農產品瑕疵檢測
* 表情辨識

---
### 教材來源
此教材由該作者(Chieh-Ming)統整，只提供教學或學習之目的。

* 台大電機系李弘毅教授 https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ
* 台大資工系林軒田教授 https://www.csie.ntu.edu.tw/~htlin/mooc/
* 莫凡 https://morvanzhou.github.io/
* 彭彭 https://www.youtube.com/user/padalab
* 大數軟體有限公司 https://www.youtube.com/channel/UCFdTiwvDjyc62DBWrlYDtlQ
