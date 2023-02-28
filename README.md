# 环境

对环境的要求不严格

我用的是py3.8的环境

## 库

pysoundfile：conda install -c conda-forge pysoundfile（读flac文件）

numpy, pandas, pytorch

## 注意事项

**将所有模型储存在指定的文件夹中因为state dict是不能挪位置的**

**加粗的是任务的单位**

预处理之后的所有任务都要保证数据顺序不要变

# 流程

1.  **数据预处理**
    1.  加载数据
    
    2.  切片：把声音数据切割成帧（暂定0.2s）
    
    3.  删除安静的部分（停顿）
    
    4.  （选）精简数据（训练数据中，一个人100条音频，一个音频15秒，每0.2秒一帧，一个人大概有7500条数据，总共大概180k条）
    
    5.  fft（得到特征，400维）
    
    6.  t-SNE（降维）
    
        1.  输入：dataX=#data*400维，dataY=y
    
        2.  输出：降维后的属性=#data*2~5维，dataY=y（保证顺序不要变就可以不考虑y）
    
    7.   knn（聚类）
    
         1.  输入：dataX=#data*400维，dataY=y
    
         2.  定义大类叫做specialty
    
         3.  输出：class->specialty的映射，dataX对应的specialty的label
    
         4.  写入本地文件。随便选格式，同时写一下读取接口
    
2.  **设计骨干+专家架构**（（pytorch）

    1.  暂定为通用
    2.  全连接网络：400, #out_dim, #out_dim, #out_dim, #out_dim, #out_dim
    3.  激活函数是tanh，最后一层softmax
    4.  输入大小是#data*400
    5.  输出大小是#data

3.  训练（pytorch）

    1.  **骨干**（写用于训练和使用的代码）
        1.  训练：所有label都是specialty label
            1.  输入：train_data, train_label, valid_data, valid_label
            2.  输出：模型（用pytorch的pt格式存state dict），每个epoch的准确率和loss（展示用）
        2.  使用
            1.  输入：test_data
            2.  输出：specialty_label，把样本和标签分成每个specialty对应的子集（即[子集1，子集2....]）下面两种情况都要做
                1.  第一种情况：子集i=[data, class_label]
                2.  第二种情况：子集i=[data]
    2.  **专家**（写用于训练和使用的代码）
        1.  训练
            1.  输入：子集i=[data, class_label]
            2.  输出：模型（用pytorch的pt格式存state dict），每个epoch的准确率和loss（展示用）
        2.  使用
            1.  输入：子集i=[data]
            2.  输出：class_label
    3.  **整合代码**（并进行训练）
        1.  把所有专家的的class_label结合起来算整体的accuracy
        2.  搭建整体流程
            1.  输入：老师给的数据集文件
            2.  输出：预测class_label
        3.  对整体模型再次进行训练以微调参数
            1.  输入：老师给的数据集文件
            2.  输出：所有模型文件