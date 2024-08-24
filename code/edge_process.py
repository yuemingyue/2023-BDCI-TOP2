import pandas as pd

node=pd.read_csv("data/raw_data/train_90.csv")
edge=pd.read_csv("data/raw_data/edge_90.csv")

#A榜测试集处理
test_edge=pd.read_csv("data/raw_data/edge_test_4_A.csv")
#找出一个月内出现过的边
train_edge=edge[edge['date_id'].isin([x for x in range(20230303,20230332)])]
last_30_train_edge=train_edge[['geohash6_point1','geohash6_point2']].drop_duplicates()
test_egdes=test_edge[['geohash6_point1','geohash6_point2']].drop_duplicates()
common_edges=pd.merge(last_30_train_edge,test_egdes,'inner')
processed_test_edge=pd.merge(common_edges,test_edge,'inner')
#保存文件
processed_test_edge.to_csv("data/user_data/processed_edge/edge_test_4_A1.csv",index=0)
print("A榜边集处理完成！")
#B榜测试集处理
test_edge=pd.read_csv("data/raw_data/edge_test_3_B.csv")
#找出一个月内出现过的边
train_edge=edge[edge['date_id'].isin([x for x in range(20230303,20230332)])]
last_30_train_edge=train_edge[['geohash6_point1','geohash6_point2']].drop_duplicates()
test_egdes=test_edge[['geohash6_point1','geohash6_point2']].drop_duplicates()
common_edges=pd.merge(last_30_train_edge,test_egdes,'inner')
processed_test_edge=pd.merge(common_edges,test_edge,'inner')
#保存文件
processed_test_edge.to_csv("data/user_data/processed_edge/edge_test_3_B1.csv",index=0)
print("B榜边集处理完成！")