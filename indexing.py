# -*- coding: utf-8 -*-
# Author: zhleternity

import init_paths
import argparse
from corefuncs.feature_extract.BinHashExtractor import BinHashExtractor
from corefuncs.feature_extract.DeepFeaturesExtractor import DeepFeaturesExtractor
from corefuncs.config import *
from corefuncs.indexer.DeepIndexer import DeepIndexer
import scipy.io as sio
from PIL import Image


# path to model file
model_file = "examples/facescrub/models/eternity_face_48_iter_325.caffemodel"#"examples/foods25/foods25_48.caffemodel"
# path to features prototxt
feat_proto = "examples/facescrub/deploy_fc7.prototxt"
# path to binary hash code prototxt
bin_proto = "examples/facescrub/face_48_deploy.prototxt"


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--use_gpu", help='define GPU use',
                default=1, type=int)
ap.add_argument("-d", "--dataset_dir", required=True,
                help="input data images list")
ap.add_argument("-o", "--output", required=True,
                help="deepfeatures db")
args = vars(ap.parse_args())

use_gpu = 0
if args["use_gpu"]:
    use_gpu = 1

# initialize the deep feature indexer
di = DeepIndexer(args["output"], estNumImages=cfg.FE.NUM_IM_ESTIMATED,
                 maxBufferSize=cfg.FE.MAX_BUF_SIZE, verbose=True)

'''
    48-bits binary codes extraction
'''

binHash_extractor = BinHashExtractor(use_gpu, cfg.FE.BINARY_CODE_LEN, bin_proto,
                                model_file, cfg.FE.MODEL_MEAN)
list_im = []
datasets_DIR = args['dataset_dir']
l_dataset_categ = os.listdir(datasets_DIR)
for i, dataset_categ in enumerate(l_dataset_categ):
    dataset_categ_DIR = os.path.join(datasets_DIR, dataset_categ)
    l_dataset = os.listdir(dataset_categ_DIR)
    if cfg.FE.CHECK_DS == 1:
        print ('[{}/{}] Loading & checking dataset_categ: {}'.format(i + 1, len(l_dataset_categ), dataset_categ))

    for dataset_fn in l_dataset:
        im_pn = os.path.join(dataset_categ_DIR, dataset_fn)
        if not os.path.isdir(im_pn):
            try:
                if cfg.FE.CHECK_DS == 1:
                    im_test = Image.open(im_pn)
                    del im_test
                list_im.append(im_pn)
            except:
                os.remove(im_pn)
                print ('index_binary48_features> IOError: cannot identify image file')
                print ('Wrong file deleted: {}'.format(im_pn))

binary_codes = binHash_extractor.extract(list_im, cfg.FE.MODEL_BIN_LAYER)

# save binary codes
if cfg.FE.SAVE_IN_MAT_FILE == 1:
    sio.savemat(cfg.FE.BIN_CODE_MAT_FILE, mdict={'binary_codes': binary_codes, list_im: 'list_im'})

'''
    layer7 feature extraction
'''
df_extractor = DeepFeaturesExtractor(use_gpu, cfg.FE.FEATURE_VECTOR_LEN, feat_proto,
                                model_file, cfg.FE.MODEL_MEAN)

df_l7 = df_extractor.extract(list_im, cfg.FE.MODEL_FEAT_LAYER)

# save binary codes
if cfg.FE.SAVE_IN_MAT_FILE == 1:
    sio.savemat(cfg.FE.DEEP_FEAT_MAT_FILE, mdict={'feat_test': df_l7, 'list_im': list_im})

"""
    Indexing binary code && deep features in hdf5 system
"""
for i, (feature_vector, binary_code) in enumerate(zip(df_l7, binary_codes)):
    # check to see if progress should be displayed
    if i > 0 and i % 10 == 0:
        di._debug("saved {} images".format(i), msgType="[PROGRESS]")

    im_pn = list_im[i]
    filename = im_pn[im_pn.rfind(os.path.sep) + 1:]

    # index the features & binary code
    di.add(filename, feature_vector, binary_code)

# finish the indexing process
di.finish()
