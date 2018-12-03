#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 


# Read data from file 'filename.csv'
df = pd.read_csv("./data/sales_train.csv") 

#Drop columns which doesnt have any impact on training and testing
excludeColumns =['date_block_num', 'date']

df.drop(excludeColumns, axis=1, inplace=True)

i = 0
def appendCategoryIdToDF(eachDFRow):
    item_category_id = items[items['item_id'] == eachDFRow['item_id']].iloc[0]['item_category_id']
    #print("before",item_category_id)
    groupedCategoryID = item_categories[item_categories['item_category_id'] == item_category_id].iloc[0]['category_id']
    #print("after",groupedCategoryID)
    eachDFRow['category_id'] = groupedCategoryID
    print(df[:5])
    return eachDFRow

zeroColumns = ['item_cnt_day','item_price']

#Replace negative values in columns to zero
print('Before:')
print(sum(n < 0 for n in df['item_cnt_day']))
for col in zeroColumns:
    df[df[col] < 0] = 0
print('After:')
print(sum(n < 0 for n in df['item_cnt_day']))

# =============================================================================
# read sales_train dataframe df and update the category_id in sales_train 
# dataframe df as per the new grouped category_id in item_categories.csv 
# =============================================================================
item_categories = pd.read_csv("./data/item_categories.csv")  
items = pd.read_csv("./data/items.csv") 
# =============================================================================
# Call appendCategoryIdToDF() to generate the category id for additional inference
# This method will create a category id through Category grouping process
# =============================================================================
df = df.apply(appendCategoryIdToDF,axis=1)
 


