# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:30:06 2018

@author: Surya
"""

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import time
#from keras.models import Sequential
#from keras.layers import Dense,Activation,Flatten
from math import sqrt
from numpy import loadtxt
from itertools import product
from sklearn import preprocessing
from xgboost import plot_tree



df_item_categories=pd.read_csv('./data/item_categories.csv')
df_items=pd.read_csv('./data/items.csv')
df_sales_train=pd.read_csv('./data/sales_train.csv')
df_shops=pd.read_csv('./data/shops.csv')
df_test=pd.read_csv('./data/test.csv')
df_sample_submission = pd.read_csv('./data/sample_submission.csv')

zeroColumns = ['item_cnt_day']

#Replace negative values in columns to zero
for col in zeroColumns:
    df_sales_train.loc[df_sales_train[col] < 0, col] = 0


#trying to get shop_id and items for each month,build new dataset
#get a list of all possible combinations of shops,items and then merge it with the grouped data
#to get all possible valid combinations for each month
    
all_combinations=[]
for b in df_sales_train['date_block_num'].unique():
#    get all unique shops for this date block
    all_shops=df_sales_train[df_sales_train['date_block_num']==b]['shop_id'].unique()
#    get all items for this unique date block
    all_items=df_sales_train[df_sales_train['date_block_num']==b]['item_id'].unique()
#    to get all combinations of these 2, doing int32 to improve speed of calculations
#    getting cant merge on np.array use np.series instead error so using try catch
    required_columns=['shop_id','item_id','date_block_num']
    try:
        all_combinations.append(np.array(list(product(*[all_shops,all_items,[b]]))))   
#    select only desired columns and making a dataframe for it
        all_combinations=pd.DataFrame(np.vstack(all_combinations),columns=required_columns)
    except:
        pass
    
#grouping the given data getting monthly sales by summing 
groups=df_sales_train.groupby(['shop_id','item_id','date_block_num'],as_index=True)
df_final=groups.agg({'item_cnt_day':'sum','item_price':'mean'}).reset_index()
df_final=df_final.rename(columns={'item_cnt_day':'item_cnt_month'})
df_final=df_final.rename(index=str,columns={'item_cnt_day':'item_cnt_month'})
# merge all combinations and grouped data
merged_set=pd.merge(df_final,all_combinations,on=required_columns,how='left')
# adding category to this merged table
merged_set=pd.merge(merged_set,df_items[['item_id','item_category_id']],on='item_id')

#taking out required important features
features_in_output=['shop_id','item_id','item_category_id','date_block_num','item_cnt_month']
train=merged_set[features_in_output]
#convert month sales to integer value
train.item_cnt_month=train.item_cnt_month.astype(int)


#train test spitting
t_train=train[train.date_block_num<33]
t_test=train[train.date_block_num==33]

#getting x and y for train and test data
X_train=t_train.iloc[:,0:4]
Y_train=t_train.iloc[:,4]

X_test=t_test.iloc[:,0:4]
Y_test=t_test.iloc[:,4]


print('starting xg boost')
t1=time.time()
#trying with xg boost
model = xgb.XGBRegressor(max_depth = 10, min_child_weight=0.5, subsample = 1, eta = 0.3, num_round = 1000, seed = 1)
model.fit(X_train, Y_train, eval_metric='rmse')
y_pred = model.predict(X_test)
y_pred=pd.Series(y_pred)
y_pred=np.rint(y_pred)
y_pred=y_pred.astype(int)
t2=time.time()
print('time taken to run xg boost is '+str( ((t2-t1)/60))+ ' minutes')
print('The rmse of prediction is:', mean_squared_error(Y_test, y_pred) ** 0.5)


print('starting nn')
t1=time.time()
NN_model = Sequential()
  # The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))
  # The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
 # Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()
NN_model.fit(X_train, Y_train, epochs=1, batch_size=10, validation_split = 0.2)
predictions = NN_model.predict(X_test)
t2=time.time()
print('time taken to run nn is '+str( ((t2-t1)/60))+ ' minutes')



#linear regression
print('starting linear regression')
t1=time.time()
lin_regr = linear_model.LinearRegression()

# Train the model using the training sets
lin_regr.fit(X_train, Y_train)

# Make predictions using the testing set
lin_regr_y_pred = lin_regr.predict(X_test)
t2=time.time()
print('time taken to run linear regression  is '+str( ((t2-t1)/60))+ ' minutes')
# The coefficients
print('Coefficients: \n', lin_regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, lin_regr_y_pred)**0.5)


from sklearn import ensemble
print('starting gbr')
t1=time.time()
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
gbr = ensemble.GradientBoostingRegressor(**params)
gbr.fit(X_train, Y_train)
gbr_y_pred=gbr.predict(X_test)
t2=time.time()
print('time taken to run gbr is '+str( ((t2-t1)/60))+ ' minutes')
print('The rmse of prediction is:', mean_squared_error(Y_test, gbr_y_pred) ** 0.5)
#print('accuracy with gbr is '+ str(accuracy_score(Y_test,gbr_y_pred)))


import lightgbm as lgb
print('starting lgbm')
t1=time.time()
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

# predict
lgbm_y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
t2=time.time()
print('time taken to run lgbm is '+str( ((t2-t1)/60))+ ' minutes')
print('The rmse of prediction is:', mean_squared_error(Y_test, lgbm_y_pred) ** 0.5)


from sklearn.ensemble import RandomForestRegressor
print('starting random forest')
randomForest = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
t1=time.time()
randModel=randomForest.fit(X_train, Y_train)
y_pred_rf = randModel.predict(X_test)
t2=time.time()
print('time taken to Random Forest is '+str( ((t2-t1)/60))+ ' minutes')
print("Mean squared error: %.2f"  % mean_squared_error(Y_test,y_pred_rf))



