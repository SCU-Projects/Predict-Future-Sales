#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 


# =============================================================================
# # Read data from file 'filename.csv'
#df = pd.read_csv("./data/sales_train.csv")
# 
# #Drop columns which doesnt have any impact on training and testing
# excludeColumns =['date_block_num', 'date']
# 
# df.drop(excludeColumns, axis=1, inplace=True)
# 
# i = 0
# def appendCategoryIdToDF(eachDFRow):
#     item_category_id = items[items['item_id'] == eachDFRow['item_id']].iloc[0]['item_category_id']
#     #print("before",item_category_id)
#     groupedCategoryID = item_categories[item_categories['item_category_id'] == item_category_id].iloc[0]['category_id']
#     #print("after",groupedCategoryID)
#     eachDFRow['category_id'] = groupedCategoryID
#     return eachDFRow
# 
# zeroColumns = ['item_cnt_day','item_price']
# 
# #Replace negative values in columns to zero
# print('Before:')
# print(sum(n < 0 for n in df['item_cnt_day']))
# for col in zeroColumns:
#     df.loc[df[col] < 0, col] = 0
#     
# print('After:')
# print(sum(n < 0 for n in df['item_cnt_day']))
# =============================================================================

# =============================================================================
# We have gropued this new category id as category_id in sales_train.csv,
# hence we dont need to  uncomment this block, because it will take 1 hour to 
# complete, lets use the category_id which is added to the sales_train.csv datase
# # =============================================================================
# # read sales_train dataframe df and update the category_id in sales_train 
# # dataframe df as per the new grouped category_id in item_categories.csv 
# # =============================================================================
#item_categories = pd.read_csv("./data/item_categories.csv")  
#items = pd.read_csv("./data/items.csv") 
# # =============================================================================
# # Call appendCategoryIdToDF() to generate the category id for additional inference
# # This method will create a category id through Category grouping process
# # =============================================================================
#df = df.apply(appendCategoryIdToDF,axis=1)
#df.to_csv("./data/grouped_sales_train.csv")
# =============================================================================
# Now we have grouped, hence lets use grouped_Sales_train.csv for use, 
# this is same as the sales_train but the diff is it has re-phrased category_id

df = pd.read_csv("./data/grouped_sales_train.csv")
df.head()