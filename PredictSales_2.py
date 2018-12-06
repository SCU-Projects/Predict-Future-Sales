# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:19:05 2018

@author: Surya
"""
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
#from translate import Translator


df_item_categories=pd.read_csv('./data/item_categories.csv')
df_items=pd.read_csv('./data/items.csv')
df_sales_train_v2=pd.read_csv('./data/sales_train.csv')
df_shops=pd.read_csv('./data/shops.csv')
df_test=pd.read_csv('./data/test.csv')


df_items=df_items.drop(columns=['item_name'])
df_item_categories=df_item_categories.drop(columns=['item_category_name'])
df_sales_train_v2=df_sales_train_v2.drop(columns=['date','item_price'])


df_cat_items=pd.merge(df_item_categories,df_items,on='item_category_id')


df_merge_total=pd.merge(df_sales_train_v2,df_cat_items,on='item_id')


zeroColumns = ['item_cnt_day']

 #Replace negative values in columns to zero
print('Before:')
print(sum(n < 0 for n in df_merge_total['item_cnt_day']))
for col in zeroColumns:
    df_merge_total.loc[df_merge_total[col] < 0, col] = 0
     
print('After:')
print(sum(n < 0 for n in df_merge_total['item_cnt_day']))


#trying only on sales v2
#df_final=df_sales_train_v2.groupby(['date_block_num','shop_id','item_id'],as_index=True).sum().reset_index()
#df_final=df_final.rename(index=str,columns={'item_cnt_day':'item_cnt_month'})







df_merge_total_grouped=df_merge_total.groupby(['date_block_num','shop_id','item_category_id','item_id'])['item_cnt_day'].sum().reset_index()
df_merge_total_grouped=df_merge_total_grouped.rename(index=str, columns={'item_cnt_day':'item_cnt_month'})
#df_merge_total_grouped=df_merge_total_grouped[['date_block_num','shop_id','item_category_id','item_id','item_cnt_month']]




#write output to a csv file
#df_merge_total_grouped.to_csv('./data/grouped_total.csv', encoding='utf-8', index=False,sep=',')

#get train and test data
train=df_merge_total_grouped[df_merge_total_grouped.date_block_num<33]
test=df_merge_total_grouped[df_merge_total_grouped.date_block_num==33]

X_train=train.iloc[:,0:4]
Y_train=train.iloc[:,4:]
X_test=test.iloc[:,0:4]
Y_test=test.iloc[:,4:]
#not using any standardization, split and apply everything
#beacuse all are just in numbers and we have only one quantity, item_cnt_month
#applying linear regression

# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
y_pred = pd.Series( (v[0] for v in y_pred) )

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"  % mean_squared_error(Y_test, pd.Series(y_pred)))
print('Variance score: %.2f' % r2_score(Y_test, y_pred))
#print(accuracy_score(Y_test, y_pred))










