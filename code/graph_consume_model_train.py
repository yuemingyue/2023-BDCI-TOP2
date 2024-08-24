import pandas as pd
import networkx as nx
import torch_geometric as pyg
import numpy as np
import random
import os
import torch
import dgl
from tqdm import tqdm
from torch.nn import Linear
import torch_geometric as pyg
from dgl.nn import GATConv,EGATConv,SAGEConv,SGConv,GINConv
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,r2_score

#确保可复现性
def setup_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
 
    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.
 
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
 
    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)

device=('cuda' if torch.cuda.is_available() else 'cpu')
print("开始读取数据")
#train
node=pd.read_csv("data/raw_data/train_90.csv")
edge=pd.read_csv("data/raw_data/edge_90.csv")
edge['date_id']=pd.to_datetime(edge['date_id'],format='%Y%m%d')
#test
test_node=pd.read_csv("data/raw_data/node_test_4_A.csv")
test_edge=pd.read_csv("data/user_data/processed_edge/edge_test_4_A1.csv")
test_edge['date_id']=pd.to_datetime(test_edge['date_id'],format='%Y%m%d')
#split_date
train_date=set(pd.to_datetime(node['date_id'],format='%Y%m%d').tolist())
test_date=set(pd.to_datetime(test_node['date_id'],format='%Y%m%d').tolist())
#data cat
node_data =  pd.concat([node,test_node],axis = 0).fillna(-1)
node_data = node_data.sort_values(['geohash_id','date_id']).reset_index(drop=True)
edge_data =  pd.concat([edge,test_edge],axis = 0)
edge_data = edge_data.sort_values(['geohash6_point1','date_id']).reset_index(drop=True)#提取特征

# #滑窗特征
# size = 4
feat_list=node_data.drop(['geohash_id','date_id','active_index','consume_index'],axis=1).columns.tolist()
# 移除特征为0
# feat_list.remove('F_23')
# feat_list.remove('F_27')
# size = 4
# grouped = node_data.groupby('geohash_id')
# for i in feat_list:
#     node_data[f'{i}_previous_{size}_days_volatility'] = grouped[f'{i}'].rolling(window=size).std().reset_index()[f'{i}']
# for i in feat_list:
#     node_data[f'{i}_previous_{size}_days_volatility_week'] = grouped[f'{i}'].rolling(window=size).std().reset_index()[f'{i}']*np.sqrt(5)
# for i in feat_list:
#     node_data[f'{i}_previous_{size}_days_volatility_month'] = grouped[f'{i}'].rolling(window=size).std().reset_index()[f'{i}']*np.sqrt(21)   
#日期特征
node_data['date_id']=pd.to_datetime(node_data['date_id'],format='%Y%m%d')
node_data['month']=node_data['date_id'].dt.month
node_data['day']=node_data['date_id'].dt.day
node_data['dayofweek']=node_data['date_id'].dt.dayofweek
node_data['is_weekend']=node_data['dayofweek'].map(lambda x:1 if x==5 or x==6 else 0)
#划分数据
node_data=node_data.dropna(axis=0)
train_node=node_data[node_data['date_id'].isin(train_date)]
test_node=node_data[node_data['date_id'].isin(test_date)]
print("数据划分完成！,开始建立图数据")
#按照时序构造每个图
grouped_node=node_data.sort_values(['geohash_id','date_id']).groupby('date_id')
#图列表
G_list=[]
#特征列表
for date_id,group in tqdm(grouped_node):
    group=group.sort_values('geohash_id')
    
    feature_name=group.drop(['geohash_id','date_id','active_index','consume_index'],axis=1).columns
    #建图
    G=nx.DiGraph()
    for index,row in group.iterrows():
        #用id作为图中节点名
        node_id=row['geohash_id']
        if 'active_index' in group.columns:
            G.add_node(node_id,x=list(row[feature_name].values),
                       y=list(row[['active_index','consume_index']]))
        else:
            G.add_node(node_id,x=list(row[feature_name].values))
    node_list=list(set(group['geohash_id'].tolist()))
    for eindex,erow in edge_data[edge_data['date_id']==date_id].iterrows():
        source_id=erow['geohash6_point1']
        target_id=erow['geohash6_point2']
        if source_id in node_list and target_id in node_list:
            G.add_edge(source_id,target_id,edge_attr=[erow['F_1'],erow['F_2']])
            G.add_edge(target_id,source_id,edge_attr=[erow['F_1'],erow['F_2']])
    pygraph=pyg.utils.from_networkx(G)
    G_data=dgl.from_networkx(G,node_attrs=['x'],edge_attrs=['edge_attr']).to(device)
    G_data.name=date_id
    G_data.y=pygraph.y.to(device)
    G_list.append(G_data)
print("图数据建立完成，开始图数据滑窗")
time_data=[]
#选择时序个数
look_back=20
time_test=4 
print("滑窗长度为",look_back+time_test)
for time in tqdm(range(0,len(G_list)-look_back-time_test+1)):
    #建图
    G=nx.DiGraph()
    #添加节点
    for index in range(time,time+look_back+time_test):
          G.add_node(G_list[index].name,graph=G_list[index])
    #添加边
    for index in range(time,time+look_back+time_test-1):
          G.add_edge(G_list[index].name,G_list[index+1].name,edge_attr=index)
    #转换数据
    G_data=dgl.from_networkx(G,edge_attrs=['edge_attr']) 
    G_data.graph=[G_list[index] for index in range(time,time+look_back+time_test)]
    G_data.y=torch.stack([G_list[index].y for index in range(time,time+look_back+time_test)])
    G_data=G_data.to(device)
    time_data.append(G_data)
    
train_data=time_data[:-4]
valid_data=time_data[0:4]
valid_date=[graph.name for graph in valid_data[-1].graph][-4:]
test_data=time_data[-4:]

class model2(torch.nn.Module):
    def __init__(self,num_features,res_dim):
        super().__init__()
        self.relu=nn.ELU()
        self.tanh=nn.Tanh()
        #小图
        self.conv1 = GATConv(39,32,1,activation=nn.Tanh())
        self.conv2 = GATConv(32,16,1,activation=nn.Tanh())
        self.conv3 = GATConv(16,8,1,activation=nn.Tanh())
        #大图
        self.conv5 = SAGEConv(47,32,'mean',activation=nn.Tanh())
        self.conv6 = SAGEConv(32,16,'mean',activation=nn.Tanh())
        self.conv7 = SAGEConv(16,8,'mean',activation=nn.Tanh())
        #LSTM
        self.lstm1=nn.LSTM(55,256,batch_first=True,bidirectional=False)
        #归一化
        self.norm=nn.BatchNorm1d(look_back+time_test,1e-3)
        #DNN
        self.dnn1 = nn.Sequential(
            nn.Linear(256+47,512),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(512,320),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(320,160),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(160,80),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(80,1),
        )
    
    def forward(self,data):
        #小图batch输入
        graph=dgl.batch(data.graph)
        graph = dgl.add_self_loop(graph)
        feature=torch.stack([graph.ndata['x'] for graph in data.graph])
        out=self.conv1(graph,graph.ndata['x'])
        out=self.conv2(graph,out)
        out=self.conv3(graph,out)
        out=out.view(len(data.graph),1140,-1)
        out=torch.cat([feature,out],dim=2)
        #大图
        data = dgl.add_self_loop(data)
        data.x=out
        out=self.conv5(data,out)
        out=self.conv6(data,out)
        out=self.conv7(data,out)
        #CAT
        out=torch.cat([data.x,out],dim=2)
        #LSTM
        out=out.permute(1,0,2)
        out,_=self.lstm1(out)
        out=torch.cat([data.x.permute(1,0,2),out],dim=2)
        #NORM
        out=self.norm(out)
        #DNN
        out1=self.dnn1(out)
        #PERMUTE
        out1=out1.permute(1,0,2)
        return out1

def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    mape=(abs(y_predict -y_test)/ y_test).mean()
    r_2=r2_score(y_test, y_predict)
    return rmse

def valid(flag):
    valid_node=node_data[node_data['date_id'].isin(valid_date)]
    valid_pred=[]
    for data in valid_data[-4:]:
        if flag==1:
            model.eval()
            out=model(data[:-1])
            valid_pred.append(out[-1:,:,:])
        else:
            model.eval()
            out=model(data)
            valid_pred.append(out[-1:,:,:])

    if len(valid_pred)>1:
        valid_pred=torch.stack(valid_pred,dim=1).reshape(-1,1140,1)

    else:
        valid_pred=valid_pred[0]

    grouped_valid_node=valid_node.groupby('date_id')
    index=0
    df_list=[]
    for date_id,group in grouped_valid_node:
        group=group.sort_values('geohash_id')
        res=valid_pred[index].tolist()
        if flag==1:
            group[['activity_level']]=res
        else:
            group[['consumption_level']]=res
        df_list.append(group)
        index+=1
    
    result=pd.DataFrame()
    for df in df_list:
        result=pd.concat([result,df],axis=0)
    real_res=pd.merge(valid_node,result,how='inner')
    real_res.fillna(-1,inplace=True)
    if flag==1:
        real_res['activity_level']=real_res['activity_level'].round(3)
        print("valid_acitivity:",evaluation(real_res[['activity_level']],valid_node[['active_index']]))
        consume=evaluation(real_res[['activity_level']],valid_node[['active_index']])
    else:
        real_res['consumption_level']=real_res['consumption_level'].round(2)
        print("valid_cosnumpation:",evaluation(real_res[['consumption_level']],valid_node[['consume_index']]))
        consume=evaluation(real_res[['consumption_level']],valid_node[['consume_index']])
    return real_res,consume

    
def rmse(predictions, targets):
    return torch.sqrt(F.mse_loss(predictions, targets))

def mape(predictions, targets):
    absolute_errors = torch.abs(targets - predictions)
    percentage_errors = torch.true_divide(absolute_errors, targets)
    return torch.mean(percentage_errors) * 100

for num in range(2):
    for i in range(10):
        torch.cuda.empty_cache()
    setup_seed(8019)
    model=model2(39,1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002,weight_decay=5e-4)
    loss_fn=torch.nn.MSELoss().to(device)
    print("准备开始训练模型")

    epochs=20
    train_list=[0,20,16,12,8,4]
    index=1
    scheduler = CosineAnnealingLR(optimizer,T_max=epochs,eta_min=0)
    model.train()
    for epoch in range(epochs):
        loss_avg=[]
        rmse_avg=[]
        mape_avg=[]
        i=epoch//20
        train_test=train_list[index+i]
        for data in train_data:
            optimizer.zero_grad()
            out1= model(data)
            loss1=loss_fn(out1[-train_test:,:,0],data.y[-train_test:,:,1])
            loss1.backward()
            optimizer.step()
            mape_avg.append(mape(out1[-train_test:,:,0],data.y[-train_test:,:,1]))
            rmse_avg.append(rmse(out1[-train_test:,:,0],data.y[-train_test:,:,1]))
            loss_avg.append(loss1.item())
        scheduler.step()
        res,consume=valid(2)
        loss=sum(loss_avg)/len(loss_avg)
        print(f'now epoch{epoch}:loss:{sum(loss_avg)/len(loss_avg)},rmse:{sum(rmse_avg)/len(rmse_avg)},mape:{sum(mape_avg)/len(mape_avg)}')
        if epoch==12:
            break

print("训练完成！正在保存模型")
model_save_dir="data/user_data/graph_model/graph_consume_model.pt"
torch.save(model,model_save_dir)