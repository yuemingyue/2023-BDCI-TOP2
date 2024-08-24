#导入所用包
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing, metrics
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta 
from sklearn.preprocessing import LabelEncoder,StandardScaler,KBinsDiscretizer,OneHotEncoder,PolynomialFeatures
from tqdm import tqdm
import random 
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import warnings
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
warnings.filterwarnings('ignore')

#设定种子
def seed_set(active_seed,consume_seed,mode):
    if mode == 'active':
        random.seed(active_seed)
        os.environ['PYTHONHASHSEED'] = str(active_seed)
        np.random.seed(active_seed)
        torch.manual_seed(active_seed)
        torch.cuda.manual_seed(active_seed)
        torch.cuda.manual_seed_all(active_seed)

    if mode == 'consume':
        random.seed(consume_seed)
        os.environ['PYTHONHASHSEED'] = str(consume_seed)
        np.random.seed(consume_seed)
        torch.manual_seed(consume_seed)
        torch.cuda.manual_seed(consume_seed)
        torch.cuda.manual_seed_all(consume_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    
def worker_init(worked_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
#针对active和conusme提取不同特征
from chinese_calendar import is_holiday, is_workday
def get_time_feature(df,mode):
    if mode == 'consume':
        df['date_id_'] = pd.to_datetime(df['date_id'], format='%Y%m%d')
        df['day_of_week'] = df['date_id_'].dt.dayofweek  
        df['is_weekday'] = df['date_id_'].apply(is_workday).astype(int)
        df['is_holiday'] = df['date_id_'].apply(is_holiday).astype(int)
        df['月']=df['date_id_'].apply(lambda x: x.month)
        df['日']=df['date_id_'].apply(lambda x: x.day)
        return df
    if mode =='active':
        df['date_id_'] = pd.to_datetime(df['date_id'], format='%Y%m%d')
        return df

#提取active特征
def getActiveFeat(data):
#     差分，滑窗：最大值 最小值 均值 日波动率 周波动率 月波动率 
    feat_list = [ f"F_{i+1}" for i in range(35)]
    # 移除特征为0
    feat_list.remove('F_23')
    feat_list.remove('F_27')
    size = 4  
    grouped = data.groupby('geohash_id')
    for i in feat_list:
        data[f'{i}_diff'] = grouped[f'{i}'].diff().reset_index()[f'{i}']
    for i in feat_list:
        data[f'{i}_previous_{size}_days_max'] = grouped[f'{i}'].rolling(window=size).max().reset_index()[f'{i}']
    for i in feat_list:
        data[f'{i}_previous_{size}_days_min'] = grouped[f'{i}'].rolling(window=size).min().reset_index()[f'{i}']
    for i in feat_list:
        data[f'{i}_previous_{size}_days_mean'] = grouped[f'{i}'].rolling(window=size).mean().reset_index()[f'{i}']
    for i in feat_list:
        data[f'{i}_previous_{size}_days_volatility'] = grouped[f'{i}'].rolling(window=size).std().reset_index()[f'{i}']
    for i in feat_list:
        data[f'{i}_previous_{size}_days_volatility_week'] = grouped[f'{i}'].rolling(window=size).std().reset_index()[f'{i}']*np.sqrt(5)
    for i in feat_list:
        data[f'{i}_previous_{size}_days_volatility_month'] = grouped[f'{i}'].rolling(window=size).std().reset_index()[f'{i}']*np.sqrt(21)   
    return data

#提取consume特征
def getConsumeFeat(data):
#    差分 滑窗：日波动率，周波动率，月波动率
    feat_list = [ f"F_{i+1}" for i in range(35)]
    feat_list.remove('F_23')
    feat_list.remove('F_27')
    size = 4
    grouped = data.groupby('geohash_id')
    for i in feat_list:
        data[f'{i}_diff'] = grouped[f'{i}'].diff().reset_index()[f'{i}']
    for i in feat_list:
        data[f'{i}_previous_{size}_days_volatility'] = grouped[f'{i}'].rolling(window=size).std().reset_index()[f'{i}']
    for i in feat_list:
        data[f'{i}_previous_{size}_days_volatility_week'] = grouped[f'{i}'].rolling(window=size).std().reset_index()[f'{i}']*np.sqrt(5)
    for i in feat_list:
        data[f'{i}_previous_{size}_days_volatility_month'] = grouped[f'{i}'].rolling(window=size).std().reset_index()[f'{i}']*np.sqrt(21)   
    return data

#划分成训练数据和测试数据
def splitData(data,test_node_std,valid = False,mode = 'active'):
    testBwindow = {20230408,20230409,20230410} #B榜测试集的窗口日期
    # 分割数据集   
    train_node = data[pd.notna(data['consume_index'])]  # 选择非缺失值的数据
    test_node = data[pd.isna(data['consume_index'])]  # 选择缺失值的数据
    test_node_B = data[pd.isna(data['consume_index']) & (data['date_id'].isin(testBwindow))]  # 选择有缺失值的b榜测试数据
    test_node_B = pd.merge(test_node_std[['geohash_id','date_id']],test_node_B.drop(['consume_index','active_index'],axis = 1),on = ['geohash_id','date_id'],how = 'left')
    if mode == 'active':
        train_node.fillna(0,inplace = True)
    if mode == 'consume':
        train_node.dropna(axis = 0,inplace = True)
    if valid == False:
        return train_node,test_node_B
    if valid == True:
        ds = sorted(list(set(train_node['date_id'])))
        timeWindow = ds[-4:]
        x_train = train_node[~train_node['date_id'].isin(timeWindow)]
        #验证集
        valid_node =  train_node[train_node['date_id'].isin(timeWindow)]
        return x_train,valid_node,test_node_B

#构造torch类型数据
class MyDataset(Dataset):
    def __init__(self, feat, active=None, consume=None, train=True):
        super().__init__()
        self.feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(1)
        self.train = train
        if train:
            self.active = torch.tensor(active, dtype=torch.float32).squeeze()
            self.consume = torch.tensor(consume, dtype=torch.float32).squeeze()

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, index):
        feat = self.feat[index]
        if self.train:
            active = self.active[index]
            consume = self.consume[index]
            return feat, active, consume

        return feat
    

def getDataLoader(train_node,test_node_B,batch_size):
    '''
    获取 数据加载器 传入不同batch_size是针对不同预测对象而言
    '''
    x_train =  train_node.copy()
    #活跃指数
    y_train_active = x_train['active_index'].values
    #消费指数
    y_train_consume = x_train['consume_index'].values
    x_train = x_train.drop(['geohash_id', 'date_id', 'active_index', 'consume_index','date_id_'],axis = 1).values
    test_B = test_node_B.drop(['geohash_id', 'date_id','date_id_'],axis = 1).values
    x_Train=MyDataset(x_train,y_train_active,y_train_consume)
    test_B = MyDataset(test_B,train=False)
    train_loader = DataLoader(x_Train,shuffle=False,batch_size=batch_size,num_workers=0,worker_init_fn=worker_init)
    test_loader_B = DataLoader(test_B,shuffle=False,batch_size=batch_size,num_workers=0,worker_init_fn=worker_init)
    return train_loader,test_loader_B

class active_CNN(nn.Module):
    def __init__(self):        
        super().__init__()
        # 定义一维卷积类
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3),  
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(64)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),  
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(64)
        )

        #定义全连接层
        self.fc=nn.Sequential(
            nn.Linear(64*64, 2048),
            nn.ReLU(),
            nn.Linear(2048,1)
         )
        self.layer1[0].weight = init.xavier_uniform_(self.layer1[0].weight)
        self.layer2[0].weight = init.xavier_uniform_(self.layer2[0].weight)
        self.layer1[0].bias = init.constant_(self.layer1[0].bias, 3e-4)
        self.layer2[0].bias = init.constant_(self.layer2[0].bias, 3e-4)   
        # 向前传播
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)

        return x
    
class consume_CNN(nn.Module):
    def __init__(self):        
        super().__init__()
        # 定义一维卷积类
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3),  
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(64)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),  
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(64)
        )

        #定义全连接层
        self.fc=nn.Sequential(
            nn.Linear(64*64, 2048),
            nn.ReLU(),
            nn.Linear(2048,1)
         )
        # 初始化第一个卷积层的权重
        self.layer1[0].weight = init.xavier_uniform_(self.layer1[0].weight)
        self.layer2[0].weight = init.xavier_uniform_(self.layer2[0].weight)
        self.layer1[0].bias = init.constant_(self.layer1[0].bias, 3e-4)
        self.layer2[0].bias = init.constant_(self.layer2[0].bias, 3e-4)
        # 向前传播
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        return x

'''训练'''
def train(train_loader,test_loader_B,epochs,mode):
    print("!!")
    for epoch in tqdm(range(epochs), desc="Training "+mode+" progress"):
        if mode == 'active':
            if epoch==85:
                torch.save(active_model.state_dict(),f'data/user_data/cnn_model/cnn_active_model.pth')
                break
            seed_set(active_seed = 42,consume_seed = None,mode = 'active') #设定active种子
            active_model.train()
        elif mode == 'consume':
            if epoch==38:
                torch.save(consume_model.state_dict(),f'data/user_data/cnn_model/cnn_consume_model.pth')
                break
            seed_set(active_seed = None,consume_seed = 3048,mode = 'consume') #设定种子
            consume_model.train()
            
        total_train_loss = 0
        for data in train_loader:
            #active
            if mode == 'active':
                feat, active= data[0].to(device), data[1].to(device)
                pred_active = active_model(feat)
                optimizer_active.zero_grad()
                loss = criterion(pred_active.squeeze(), active)
                total_train_loss += loss.item()
                loss.backward()
                optimizer_active.step()
                
            #consume
            elif mode == 'consume':
                feat, consume= data[0].to(device), data[2].to(device)
                pred_consume = consume_model(feat)
                optimizer_consume.zero_grad()
                loss = criterion(pred_consume.squeeze(), consume)
                total_train_loss += loss.item()
                loss.backward()
                optimizer_consume.step()      
        swalist =  [ epochs - i for i in range(3)]      
        # 学习率衰减
        if mode == 'active':
            active_scheduler.step()
        if mode == 'consume':
            consume_scheduler.step()
        
        avg_loss = total_train_loss / len(train_loader)
        rmse = np.sqrt(avg_loss)
        
        print(f"trainLoss: {avg_loss:.5f},rmse:{rmse}")

    #读取数据
def readData():
    train_node = pd.read_csv('data/raw_data/train_90.csv')
    test_node_A = pd.read_csv('data/raw_data/node_test_4_A.csv')
    test_node_B = pd.read_csv('data/raw_data/node_test_3_B.csv')
    data =  pd.concat([train_node, test_node_A,test_node_B],axis = 0)
    data = data.groupby('geohash_id').apply(lambda x: x.sort_values('date_id')).reset_index(drop=True)
    return train_node,test_node_A,test_node_B,data

if __name__=="__main__":

    #active
    train_node,test_node_A,test_node_B,data = readData()
    print("开始构造active数据")
    active_data = get_time_feature(data,'active')
    active_data = getActiveFeat(active_data)
    train_node,test_node_active_B = splitData(active_data,test_node_B)
    active_train_loader,active_test_loader_B = getDataLoader(train_node,test_node_active_B,batch_size = 90)
    print("active数据构造完成")
    device = 'cuda:0'
    lr = 0.0004
    criterion = torch.nn.MSELoss().to(device) #mse
    active_model = active_CNN().to(device)
    optimizer_active = torch.optim.Adam(active_model.parameters(), lr=lr,weight_decay=1e-4)
    active_scheduler = CosineAnnealingWarmRestarts(optimizer_active, T_0=1, T_mult=4, eta_min=1e-5)
    seed_set(active_seed = 42,consume_seed = None,mode = 'active')
    
    #consume
    train_node,test_node_A,test_node_B,data = readData()
    print("开始构造consume数据")
    consume_data = get_time_feature(data,'consume')
    consume_data = getConsumeFeat(consume_data)
    train_node,test_node_consume_B = splitData(consume_data,test_node_B)
    consume_train_loader,consume_test_loader_B = getDataLoader(train_node,test_node_consume_B,batch_size = 16)
    print("consume数据构造完成")
    # 获取数据加载器
    lr = 0.0001
    criterion = torch.nn.MSELoss().to(device) #mse
    consume_model = consume_CNN().to(device)
    optimizer_consume = torch.optim.Adam(consume_model.parameters(), lr = lr,weight_decay=1e-4)
    consume_scheduler = CosineAnnealingWarmRestarts(optimizer_consume, T_0=1, T_mult=4, eta_min=1e-5)
    seed_set(active_seed = None,consume_seed = 3048,mode = 'consume')
    
    #pred
    print("开始预测")
    active_model = active_CNN().to(device)
    consume_model = consume_CNN().to(device)
    active_model.load_state_dict(torch.load('data/user_data/cnn_model/cnn_active_model.pth'))
    consume_model.load_state_dict(torch.load('data/user_data/cnn_model/cnn_consume_model.pth'))
    
    pred = {'consumption_level': [],'activity_level': []}
    for feat in consume_test_loader_B:
        pred_consume = consume_model(feat.to(device))    
        pred['consumption_level'] += pred_consume.squeeze().cpu().detach().numpy().tolist()

    for feat in active_test_loader_B:
        pred_active = active_model(feat.to(device))
        pred['activity_level'] += pred_active.squeeze().cpu().detach().numpy().tolist()
    pred = pd.DataFrame(pred)
    pre = pd.concat([test_node_B['geohash_id'],pred,test_node_B['date_id']],axis=1)
    pre.to_csv('data/user_data/prediction/cnn_test_b_pred.csv', index=False, sep='\t')
    print("预测结束")
    import pandas as pd
    import networkx as nx
    import torch_geometric as pyg
    import numpy as np
    import random
    import os
    import torch
    import dgl
    from tqdm import tqdm
    import torch
    import torch.nn.functional as F
    from torch.nn import Linear
    import torch_geometric as pyg
    from dgl.nn import GATConv,EGATConv,SAGEConv,SGConv,GINConv
    import torch.nn as nn
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

    setup_seed(1132)
    device=('cuda' if torch.cuda.is_available() else 'cpu')

    #train
    node=pd.read_csv("data/raw_data/train_90.csv")
    edge=pd.read_csv("data/raw_data/edge_90.csv")
    edge['date_id']=pd.to_datetime(edge['date_id'],format='%Y%m%d')
    #test
    test_node=pd.read_csv("data/raw_data/node_test_4_A.csv")
    test_edge=pd.read_csv("data/user_data/processed_edge/edge_test_4_A1.csv")
    test_edge['date_id']=pd.to_datetime(test_edge['date_id'],format='%Y%m%d')
    #test_b
    test_b_node=pd.read_csv("data/raw_data/node_test_3_B.csv")
    test_b_edge=pd.read_csv("data/user_data/processed_edge/edge_test_3_B1.csv")
    test_b_edge['date_id']=pd.to_datetime(test_b_edge['date_id'],format='%Y%m%d')
    #split_date
    train_date=set(pd.to_datetime(node['date_id'],format='%Y%m%d').tolist())
    test_date=set(pd.to_datetime(test_node['date_id'],format='%Y%m%d').tolist())
    test_b_date=set(pd.to_datetime(test_b_node['date_id'],format='%Y%m%d').tolist())
    #data cat
    node_data =  pd.concat([node,test_node,test_b_node],axis = 0).fillna(-1)
    node_data = node_data.sort_values(['geohash_id','date_id']).reset_index(drop=True)
    edge_data =  pd.concat([edge,test_edge,test_b_edge],axis = 0)
    edge_data = edge_data.sort_values(['geohash6_point1','date_id']).reset_index(drop=True)

    #提取特征
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
    test_b_node=node_data[node_data['date_id'].isin(test_b_date)]

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

    time_data=[]
    #选择时序个数
    look_back=20
    time_test=4                                    
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

    train_data=time_data[:-7]
    valid_data=time_data[0:4]
    valid_date=[graph.name for graph in valid_data[-1].graph][-4:]
    test_data=time_data[-3:]
    del time_data

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

    class model1(torch.nn.Module):
        def __init__(self,num_features,res_dim):
            super().__init__()
            self.relu=nn.ELU()
            self.tanh=nn.Tanh()
            #小图
            # self.ginconv = GINConv(None,'mean')
            self.conv1 = GATConv(39,32,1)
            self.conv2 = GATConv(32,16,1)
            self.conv3 = GATConv(16,8,1)
            self.conv4 = GATConv(8,4,1)
            #大图
            self.conv5 = SAGEConv(43,32,'pool')
            self.conv6 = SAGEConv(32,16,'pool')
            self.conv7 = SAGEConv(16,8,'pool')
            #LSTM
            self.lstm1=nn.LSTM(51,512,batch_first=True,bidirectional=False)
            self.lstm2=nn.LSTM(512,256,batch_first=True,bidirectional=True)
            #归一化
            self.norm=nn.BatchNorm1d(24,1e-3)
            #DNN
            self.dnn1 = nn.Sequential(
                nn.Linear(512+43,1024),
                nn.Mish(),
                nn.Dropout(0.2),
                nn.Linear(1024,512),
                nn.Mish(),
                nn.Dropout(0.2),
                nn.Linear(512,256),
                nn.Mish(),
                nn.Dropout(0.2),
                nn.Linear(256,128),
                nn.Mish(),
                nn.Linear(128,64),
                nn.Mish(),
                nn.Linear(64,1),
            )

        def forward(self,data):
            #小图batch输入
            graph=dgl.batch(data.graph)
            graph = dgl.add_self_loop(graph)
            feature=torch.stack([graph.ndata['x'] for graph in data.graph])
            out=self.conv1(graph,graph.ndata['x'])
            out=self.tanh(out)
            out=self.conv2(graph,out)
            out=self.tanh(out)
            out=self.conv3(graph,out)
            out=self.tanh(out)
            out=self.conv4(graph,out)
            out=out.view(len(data.graph),1140,-1)
            out=torch.cat([feature,out],dim=2)
            #大图
            data = dgl.add_self_loop(data)
            data.x=out
            out=self.conv5(data,out)
            out=self.tanh(out)
            out=self.conv6(data,out)
            out=self.tanh(out)
            out=self.conv7(data,out)
            #CAT
            out=torch.cat([data.x,out],dim=2)
            #CNN
            # out=out.permute(1,2,0)
            # out=out.permute(2,0,1)
            #LSTM
            out=out.permute(1,0,2)
            out,_=self.lstm1(out)
            out,_=self.lstm2(out)
            out=torch.cat([data.x.permute(1,0,2),out],dim=2)
            #NORM
            out=self.norm(out)
            #DNN
            out1=self.dnn1(out)
            #PERMUTE
            out1=out1.permute(1,0,2)
            return out1
    def rmse(predictions, targets):
        return torch.sqrt(F.mse_loss(predictions, targets))

    def mape(predictions, targets):
        absolute_errors = torch.abs(targets - predictions)
        percentage_errors = torch.true_divide(absolute_errors, targets)
        return torch.mean(percentage_errors) * 100

    model1=torch.load('data/user_data/trained_model/graph_active_model.pt')
    model1.to(device)
    model2=torch.load('data/user_data/trained_model/graoh_consume_model.pt')
    model2.to(device)

    active_pred=[]
    for index in range(3):
        model1.eval()
        out1=model1(test_data[index])
        out1=(0.03*out1[-4]+0.04*out1[-3]+0.08*out1[-2]+0.85*out1[-1]).reshape(1,1140,-1)
        active_pred.append(out1[-1:,:,0])

    consume_pred=[]
    for index in range(3):
        model2.eval()
        out2=model2(test_data[index])
        out2=(0.0*out2[-4]+0.0*out2[-3]+0.0*out2[-2]+1*out2[-1]).reshape(1,1140,-1)
        consume_pred.append(out2[-1:,:])


    grouped_test_node=pd.read_csv("data/raw_data/node_test_3_B.csv").groupby('date_id')
    index=0
    df_list=[]
    for date_id,group in grouped_test_node:
        group=group.sort_values('geohash_id')
        group['activity_level']=active_pred[index][0].tolist()
        group['consumption_level']=consume_pred[index][0].tolist()
        df_list.append(group)
        index+=1

    result=pd.DataFrame()
    for df in df_list:
        result=pd.concat([result,df],axis=0)
    real_res=pd.merge(pd.read_csv("data/raw_data/node_test_3_B.csv"),result,how='inner')
    real_res['consumption_level']=real_res['consumption_level'].map(lambda x:x[0]).round(2)
    real_res['activity_level']=real_res['activity_level'].round(3)

    real_res[['geohash_id','consumption_level','activity_level','date_id']].to_csv("data/user_data/prediction/best_graph_test_b_pred.csv",index=0,sep='\t')

    import pandas as pd
    res1=pd.read_csv('data/user_data/prediction/best_graph_test_b_pred.csv',sep='\t')
    res2=pd.read_csv('data/user_data/prediction/best_cnn_test_b_pred.csv',sep='\t')
    #开始加权
    #0.5 0.5
    res3=res1.copy()
    res3[['consumption_level','activity_level']]=res1[['consumption_level','activity_level']]*0.5+res2[['consumption_level','activity_level']]*0.5
    #0.6 0.4
    res4=res1.copy()
    res4[['consumption_level','activity_level']]=res1[['consumption_level','activity_level']]*0.6+res2[['consumption_level','activity_level']]*0.4
    #0.4 0.6
    res5=res1.copy()
    res5[['consumption_level','activity_level']]=res1[['consumption_level','activity_level']]*0.4+res2[['consumption_level','activity_level']]*0.6
    #0.3 0.7
    res6=res1.copy()
    res6[['consumption_level','activity_level']]=res1[['consumption_level','activity_level']]*0.3+res2[['consumption_level','activity_level']]*0.7
    #0.7 0.3
    res7=res1.copy()
    res7[['consumption_level','activity_level']]=res1[['consumption_level','activity_level']]*0.7+res2[['consumption_level','activity_level']]*0.3
    #0.55 0.45
    res8=res1.copy()
    res8[['consumption_level','activity_level']]=res1[['consumption_level','activity_level']]*0.55+res2[['consumption_level','activity_level']]*0.45
    #0.45 0.55
    res9=res1.copy()
    res9[['consumption_level','activity_level']]=res1[['consumption_level','activity_level']]*0.45+res2[['consumption_level','activity_level']]*0.55
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error,r2_score

    def evaluation(y_test, y_predict):
        mae = mean_absolute_error(y_test, y_predict)
        mse = mean_squared_error(y_test, y_predict)
        rmse = np.sqrt(mean_squared_error(y_test, y_predict))
        mape=(abs(y_predict -y_test)/ y_test).mean()
        r_2=r2_score(y_test, y_predict)
        return rmse


    ans_list=[res1,res2,res3,res4,res5,res6,res7,res8,res9]
    from tqdm import tqdm

    print("正在进行投票融合！")
    ans=[]
    group=res1.groupby('geohash_id')
    for geohash_id,group in tqdm(group):
        rmse_list=[]
        for x1 in ans_list:
            x1_rmse=[]
            x1=x1[x1['geohash_id']==geohash_id]
            for x2 in ans_list:
                x2=x2[x2['geohash_id']==geohash_id]
                x1_rmse.append(evaluation(x1[['consumption_level','activity_level']],x2[['consumption_level','activity_level']]))
            rmse_list.append(sum(x1_rmse))
        index=rmse_list.index(min(rmse_list))
        ans.append(ans_list[index][ans_list[index]['geohash_id']==geohash_id].values)

    res=[]
    for geohash_id in ans:
        geohash_id=geohash_id.tolist()
        for x in geohash_id:
            res.append(x)
    print("融合结束！")
    res=pd.DataFrame(res)
    res.columns=res1.columns
    merge=res1[['geohash_id','date_id']]
    res=pd.merge(merge,res)
    res[['geohash_id','consumption_level','activity_level','date_id']].to_csv("data/prediction_result/b_result.csv",index=0,sep='\t')

    
    