# Artificial-Intelligence

![GITHUB](https://www.edntaiwan.com/wp-content/uploads/sites/6/images/7447e8f3-110b-4a56-82e9-9cf540145331.gif)

## Class 01 人工智慧 Artificial-Intelligence [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/01_Artificial-Intelligence/01_%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7.pdf)
* 人工智慧發展史(The History of Artificial Intelligence)
* 人工智慧的任務(The Basic Objective of Artificial Intelligence)
  * 回歸(Regression)
  * 分類(Classification)
  * 分群(Cluster)
  * 複雜性任務
* 人工智慧的學習方法
  * 非監督式學習(Unsupervised Learning)
  * 監督式學習(Supervised Learning)
  * 強化學習(Reinforcement Learning)
* Google Colab

---
## Class 02 機器學習 - 線性回歸 Linear-Regression [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/02_Linear-Regression/02_%E7%B7%9A%E6%80%A7%E5%9B%9E%E6%AD%B8.pdf)
* 參數與超參數(Parameters and Hyperparameters)
* 損失函數(Loss function): 均方誤差 (Mean-Square Error,MSE)
* 梯度下降法(Gradient Descent, GD)
* 學習速率(Learning Rate)
* 批量梯度下降法(Batch Gradient Descent, BGD)
* 隨機梯度下降法(Stochastic Gradient Descent, SGD)
* 小批量梯度下降法(Mini-batch Gradient Descent, MBGD)
* 線性回歸實作
* **Homework 01 身體質量指數預測**

---
## Class 03 機器學習 - 邏輯回歸 Logistic Regression [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/03_Logistic-Regression/03_%E9%82%8F%E8%BC%AF%E5%9B%9E%E6%AD%B8.pdf)
* 邏輯/乙狀函數(Logistic / Sigmoid Function)
* 最大似然函數(Maximum Likelihood Function)
* 勝算比(Odds Ratio)
* 正則化(Regularization)
* 邏輯回歸實作
* **Homework 02-1 數字手寫辨識**
* **Homework 02-2 糖尿病預測**

---
## Class 04 機器學習 - 決策樹與隨機森林 Decision Tree and Random Forest [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/04_Decision-Tree%26Random-Forest/04_%E6%B1%BA%E7%AD%96%E6%A8%B9%E8%88%87%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97.pdf)
* 決策樹架構
  * 分支標準：交叉熵與基尼不純度(Cross-Entropy and Gini Impurity)
  * 最大訊息增益量(Information Gain)
  * 剪枝(Pruning)
    * 預剪枝(Pre-pruning)
    * 後剪枝(Post-pruning)
* 隨機森林架構
  * 集成學習(Ensemble Learning)：裝袋(Bagging)、取樣(Sampling)
* 決策樹與隨機森林實作
* **Homework 03 字母手寫辨識**

---
## Class 05 機器學習 - 自適應增強 AdaBoost [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/05_AdaBoost/05_%E8%87%AA%E9%81%A9%E6%87%89%E5%A2%9E%E5%BC%B7.pdf)
* 提升(Boosting)
* 自適應增強(Adaptive Boosting)
* 集成學習：重新加權訓練集 (Re-weighting Training Data)
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
* **Homework **

---
## Class 07 深度學習 - 卷積神經網路 Convolutional Neural Network [[PDF]](https://github.com/j82887/Artificial-Intelligence/blob/master/07_Convolutional-Neural-Network/07_%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF.pdf)
* 卷積層(Convolution Layer)
  * 濾波器/卷積核(Filter/Convolution Kernel)
  * 遮罩/窗口(Mask)
  * 卷積/旋積(Convolution)
  * 填充(Padding)
  * 特徵圖(Feature Map)
* 池化層(Pooling Layer)
  * 局部最大池化(Local Max-pooling)
  * 局部平均池化(Local Avg-pooling)
  * 局部隨機池化(Local Ram-pooling)
  * 全域最大池化(Global Max-pooling)
  * 全域平均池化(Global Avg-pooling)
* 全連接層(Fully Connected Layer)
  * Softmax
* 自動提取特徵(Automatic Feature Extraction)
* 權重共享(Shared Weight)
* 卷積神經網路實作
* **Homework Cifar100 CNN辨識**

---
## Class 08 深度學習 - 自編碼器 Auto-Encoder [PDF]
* 編碼器(Encoder)
* 解碼器(Decoder)
* 去噪自編碼器(De-noising Auto-encoder)
* 降維(Dimension Reduction)
* 資料視覺化(Data Visualization)
* 權重初始化(Weight Initialization)

---
## Class 09 深度學習 - 生成對抗神經網路 Generative Adversarial Network [PDF]
* 生成器(Generator) 
* 鑑別器(Discriminator) 
* 結構化學習(Structured Learning)
  * Bottom-Up 
  * Top-Down

---
## Class 10 分類模型的驗證指標 Classification Model Assessment and Validation [PDF]
* 混淆矩陣(Confusion Matrix)
* 準確度(Accuracy)
* 特異度(Specificity)
* 敏感度/召回率(Sensitivity/Recall)
* 精確度(Precision)
* F1分數(F1 Score)
* ROC曲線(Receiver Operating Characteristic Curve, ROC Curve)
* PR曲線(Precision Recall Curve, PR Curve)
* 曲線下面積(Area Under Curve, AUC)
* 交叉驗證(Cross-validation, CV)
  * 分層交叉驗證(Stratified Cross-validation, SCV)
  * 留一法交叉驗證(Leave One Out Cross-validation, LOOCV)
  * 隨機分割交叉驗證(Random Cross-validation, RCV)
  * K格交叉驗證(k-fold Cross-validation)
* 學習曲線(Learning Curve)

---
## Class 11 資料前處理 - 不均勻類別問題 Class Imbalance Problems [PDF]
* 取樣(Sampling)
  * 欠取樣(Under-sampling)
  * 過取樣(Over-sampling)
* 合成少數類別過取樣技術(Synthetic Minority Over-sampling Technique)
* Tomek Link(T-Link)

---
## Class 12 資料前處理 - 特徵縮放 與 獨熱編碼 Feature Scaling and One Hot Encoding [PDF]
* 特徵縮放(Feature Scaling)
  * 重新縮放(Rescaling / Min-max Normalization)
  * 平均值正規化(Mean Normalization)
* 特徵處理/編碼(Feature Processing/Coding)
  * 排序行編碼(Sequential Encoding)
  * 獨熱編碼(One Hot Encoding)
  
---  
## Class 13 超參數搜索 Hyperparameter Search [PDF]
* 均格/網格搜索(Grid Search)
* 隨機搜索(Random Search)
* 拉丁超立方抽樣(Latin Hypercube Sampling, LHS)
* 貝葉斯優化(Bayesian Optimization)

---
## Class 14 梯度下降法的優化 Optimization of Gradient Descent Method [PDF]
* 梯度下降法(Gradient Descent)
* 動量的梯度下降法(Gradient Descent with Momentum)
* AdaGrad(Adaptive Gradient \)
* RMSProp(Root Mean Square Prop)
* Adam(Adaptive Moment Estimation)

---
## Class 15 機率模型的分類切點 [PDF]


---
### 教材來源
此教材由該作者(Chieh-Ming)統整，只提供教學或學習之目的。

* 台大電機系李弘毅教授 https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ
* 台大電機系林軒田教授 https://www.csie.ntu.edu.tw/~htlin/mooc/
* 莫凡 https://morvanzhou.github.io/
* 彭彭 https://www.youtube.com/user/padalab
* 大數軟體有限公司 https://www.youtube.com/channel/UCFdTiwvDjyc62DBWrlYDtlQ
