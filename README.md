比赛链接：https://www.datafountain.cn/competitions/979

## 一.环境依赖说明

#### 1.镜像来源：

  dockerHub:pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime;

python:3.10.9

#### 2.其他依赖：

chinese_calendar==1.9.0
networkx==3.0
numpy==1.24.2
pandas==1.3.5
scikit_learn==1.3.1
seaborn==0.13.0
tqdm==4.61.2
torch_geometric==2.4.0

dgl1.1.2+cu116

#If you have installed dgl-cuXX package, please uninstall it first.

#pip install  dgl -f https://data.dgl.ai/wheels/cu116/repo.html 
#pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

#### 3.**硬件依赖特殊说明**：

​	本次比赛中因为我们为后期融合，采取两套不同方案的。也导致两种不同系统环境。对于图方案而言dgl框架cuda限制在11.6以上。

​	时空图卷积方案：RTX A5000，cpu:19核/GPU Xeon(R) Platinum 8350c ,显存24GB；torch2.0+cuda118；python3.8

​	卷积神经网络方案：Tesla P100-PCIE-16GB;torch2.0+cuda118；python3.10.12其余依赖如上。

我们**存取了最佳模型在user_data/trained_model**里面可使用上述不同设备进行预测获得我们团队B榜最优结果。

## 二、run.sh代码说明

4.1训练cnn模型，保存模型，cnn方案预测出临时文件

```
python code/cnn_train_pred.py
```

4.2 时空图卷积方案：边处理，active指数模型训练模型保存，consume指数模型训练模型保存，时序图卷积方案预测出临时文件。

```
python code/edge_process.py
python code/graph_active_model_train.py
python code/graph_consume_model_train.py
python code/graph_pred.py
```

4.3 时空图卷积方案于卷积神经网络方案融合

```
python code/melt.py
```

4.4 最佳模型预测（**需要各方案设备完全匹配**）

由于和队友的设备和版本差异，我们纵使在分别做了可复现设置的情况下，迁移到同一设备版本下仍然与我们分别的结果存在出入，因此我们在trained_model中上传了我们当时的最优模型，并可以通过best_pred.py 集成了加载cnn和图的最优模型并进行推理和融合的过程，用于复现b榜结果。最优模型结构和与提供的代码中训练的模型结构一致。运行best_pred.py后同样会在prediction_result中生成b_result.csv代表b榜预测结果。

```
python code/best_pred.py
```





