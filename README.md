# Kaggle-Understanding-Clouds-from-Satellite-Images
## 赛题背景介绍
> 多年来，气候变化一直是我们最关心的问题，也是重要政治决策的首要问题。我们希望你能使用这次比赛的数据来帮助揭开一个重要的气候变量的神秘面纱。像马克斯·普朗克气象研究所(Max Planck Institute for meteorological)的科学家们正在领导一项关于世界上不断变化的大气的新研究，他们需要帮助来更好地理解云。<br>
`浅云`在决定地球气候方面起着巨大的作用。它们在气候模型中也很难理解和表示。通过对不同类型的云组织进行分类，马克斯·普朗克的研究人员希望提高我们对这些云的物理理解，从而帮助我们建立更好的气候模型。云的组织方式有很多种，但是不同组织形式之间的界限是模糊的。这使得构建传统的基于规则的算法来分离云特性变得很有挑战性。然而，人类的眼睛确实很擅长探测特征——比如像花一样的云在。<br>
这个挑战中，您将构建一个模型来对来自卫星图像的云组织模式进行分类。如果成功，你将帮助科学家更好地了解云将如何塑造我们未来的气候。这项研究将指导下一代模型的开发，以减少气候预测中的不确定性。
## 数据描述
* train.csv: 一共两列，第一列为Image_Label, 第二列为这个Label对应的`run length encoded segmentations`；<br>
* train_images.zip: 训练图片； <br>
* test_images.zip：测试图片; 任务是预测每个图像的4种云类型(标签)的分隔掩码. 注意: 预测的mask需要调整到`350×525`；<br>
* sample_submission.csv：提交文件格式；<br>
## 数据集划分
>根据图片包含的浅云类别数量， 利用库函数`StratifiedKFold`进行数据划分
## 模型
运用了[segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)
本比赛集成了以下两个single模型：<br>
* Unet(resnet34)(5-flod)<br>
* FPN(resnet34)<br>
##### 排名
276/1538(18%)<br>
LB:0.65521 <br>
PB:0.6485
## 文件说明
* [`Shallow_clould_CV.py`](https://github.com/hierarchyJK/Kaggle-Understanding-Clouds-from-Satellite-Images/blob/master/Shallow_clould_CV.py): 生成交叉验证数据集<br>
* [`Shallow_smp.py`](https://github.com/hierarchyJK/Kaggle-Understanding-Clouds-from-Satellite-Images/blob/master/Shallow_smp.py):训练文件<br>
* [`Shallow_clouds_interface.py`](https://github.com/hierarchyJK/Kaggle-Understanding-Clouds-from-Satellite-Images/blob/master/Shallow_clouds_interface.py): 预测推理文件<br>
* [`Exploring_Predictions.py`](https://github.com/hierarchyJK/Kaggle-Understanding-Clouds-from-Satellite-Images/blob/master/Exploring_Predictions.py): 利用验证集寻找post_process()函数中的最优的threshold和minsize<br>

### 后期同类型比赛尝试方向：
* 1、根据所有训练集搭建`classification_model`分类模型；
* 2、去除没有mask的训练集后，利用存在至少又一个mask的图片搭建`segmentation_model`分割模型；
* 3、对测试集进行推断时候，先用`classification_model`将没有mask的图片预测出来，再利用`segmentation_model`分割模型对剩下图片进行mask预测，最后将两个结果合并一下即可
