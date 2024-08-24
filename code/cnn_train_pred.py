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
    #训练模型
    print("开始训练active模型")
    train(train_loader=active_train_loader,test_loader_B=active_test_loader_B,epochs=90,mode='active')
    print("active模型训练完成")
    
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
    print("开始训练consume模型")
    train(train_loader=consume_train_loader,test_loader_B=consume_test_loader_B,epochs=50,
             mode='consume')
    print("consume模型训练完成")
    
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

    
    