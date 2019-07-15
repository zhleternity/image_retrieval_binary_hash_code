#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zhleternity
# -------------------------------------


import os
import sys


train_name = 'train.txt'
file_train = open(train_name, 'w')
val_name = 'val.txt'
file_val = open(val_name, 'w')
root_dir = "/mnt/hgfs/ubuntushared/retrieve_lsh/data/webface/imgs/" 
list = os.listdir(root_dir)
for i, l in enumerate(list):
    img_path = os.path.join(root_dir, l)
    print(img_path)
    result = img_path + '\n'
    if i <= 320000:
        file_train.write(result)
    else:
        file_val.write(result)

file_train.close()
file_val.close()
