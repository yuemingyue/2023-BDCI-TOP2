import pandas as pd

res1=pd.read_csv('data/user_data/prediction/graph_test_b_pred.csv',sep='\t')
res2=pd.read_csv('data/user_data/prediction/cnn_test_b_pred.csv',sep='\t')
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
res[['geohash_id','consumption_level','activity_level','date_id']].to_csv("data/prediction_result/result.csv",index=0,sep='\t')