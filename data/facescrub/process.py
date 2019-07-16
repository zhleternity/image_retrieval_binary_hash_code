# -*- coding: utf-8 -*-
# Author: zhleternity

import os
import sys

root_dir = "../actress/"
train_dir = "train/"
val_dir = "val/"
list = os.listdir(root_dir)
for l in list:
     people_path = os.path.join(root_dir, l, 'face')
     faces = os.listdir(people_path)
     for i, img in enumerate(faces):
         face = os.path.join(people_path, img)

         if i < int(len(faces) * 0.8):
             to_cp_name = os.path.join()
             os.system("cp face .")
         else:
             os.system("cp face .")
