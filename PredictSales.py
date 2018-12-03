#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 


# Read data from file 'filename.csv'
df = pd.read_csv("./data/sales_train.csv") 

#Drop columns which doesnt have any impact on training and testing
excludeColumns =['date_block_num', 'date']

df.drop(excludeColumns, axis=1, inplace=True)

zeroColumns = ['item_cnt_day','item_price']

#Replace negative values in columns to zero
print('Before:')
print(sum(n < 0 for n in df['item_cnt_day']))
for col in zeroColumns:
    df[df[col] < 0] = 0
print('After:')
print(sum(n < 0 for n in df['item_cnt_day']))