# -*- coding: utf-8 -*-
# Author: zhleternity

"""Set up paths"""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
caffe_root = '/home/ling/workspace/3rd/caffe/'

# Add caffe to PYTHONPATH
caffe_path = osp.join(caffe_root, 'python')
add_path(caffe_path)

# Add libs to PYTHONPATH
libs_path = osp.join(this_dir, '..', 'libs')
#add_path(libs_path)

# Add project to PYTHONPATH
proj_path = osp.join(this_dir, '..')
add_path(proj_path)
