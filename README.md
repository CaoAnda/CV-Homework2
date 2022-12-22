# CV-Homework2
 计算机视觉导论作业二

作业1：改进LeNet-5实现手写数字的分割

框架：Tensorflow, pytorch, keras, paddlepaddle，…

数据集： The Mnist Dataset of handwritten digits
http://yann.lecun.com/exdb/mnist/

预处理：对手写数字通过颜色值获得前景数字，网络搜索20张图片，通过随机切patch的方式获背景图片，将随机切的patch块与前景数字拼接获得训练样本（合成的图片，对应的分割GT）

基础网络结构 LeNet-5: http://yann.lecun.com/exdb/lenet/

任务：仿照FCN: Fully Convolutional Network 实现LeNet-5改进为手写数字分割网络

改进方式：1）Upsampling  2) Deconvolution 

分割类别：1）前景背景分割（2分类分割）  2）按照数字类别分割（10分类分割）

作业2：之前做做过的分类、分割、检测等相关任务的深度模型模型算法

提交内容：一页word总结 + 2-3缺点总结

Code + Report （不超过1页）

Due: 2023年1月6日
