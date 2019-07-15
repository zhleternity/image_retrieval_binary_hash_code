#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zhleternity
# -------------------------------------

from __future__ import print_function
import init_paths
import argparse
import cv2
import imutils
from corefuncs.search.DeepSearcher import DeepSearcher
from corefuncs.ResultsMontage import ResultsMontage
from corefuncs.feature_extract.BinHashExtractor import BinHashExtractor
from corefuncs.feature_extract.DeepFeaturesExtractor import DeepFeaturesExtractor
from corefuncs.search import dists
from corefuncs.config import *
import os
import subprocess
import numpy as np
from sklearn.neighbors import KDTree
#from layer_features import layer_features
#from extract_features import VGGNet

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
ap.add_argument("-f", "--deep_features", required=True,
                help="Path to the features & binary code database")
ap.add_argument("-d", "--dataset_dir", required=True,
                help="input data images list")
ap.add_argument("-q", "--query", required=True,
                help="Path to the query image")
args = vars(ap.parse_args())

use_gpu = 0
if args["use_gpu"]:
    use_gpu = 1

queryRelevant = []

# grap image pathname
im_paths = {}
l_categs = os.listdir(args["dataset_dir"])
for categ_name in l_categs:
    l_imgs = os.listdir(args["dataset_dir"] + '/' + categ_name)
    for im_fn in l_imgs:
        im_paths[im_fn] = os.path.join(args["dataset_dir"] + '/' + categ_name, im_fn)

'''
    48-bits binary codes extraction
'''
binHash_extractor = BinHashExtractor(use_gpu, cfg.FE.BINARY_CODE_LEN, bin_proto,
                                model_file, cfg.FE.MODEL_MEAN)

query_binarycode = (binHash_extractor.extract([args["query"]],
                                       cfg.FE.MODEL_BIN_LAYER))[0]  # 0 cause just 1 image
'''
    layer7 feature extraction
'''
df_extractor = DeepFeaturesExtractor(use_gpu, cfg.FE.FEATURE_VECTOR_LEN, feat_proto,
                                model_file, cfg.FE.MODEL_MEAN)

query_fVector = (df_extractor.extract([args["query"]], cfg.FE.MODEL_FEAT_LAYER))[0]

dSearcher = DeepSearcher(args["deep_features"], distanceMetric=dists.chi2_distance)
# compute similarities
search_result = dSearcher.search(query_binarycode, query_fVector, numResults=20, maxCandidates=100)
print("[INFO] search took: {:.2f}s".format(search_result.search_time))

# initialize the results montage
montage = ResultsMontage((240, 320), 5, 20)

# load the query image and process it
queryImage = cv2.imread(args["query"])
cv2.imshow("Query", imutils.resize(queryImage, width=320))

# loop over the individual results
for (i, (score, resultID, resultIdx)) in enumerate(search_result.results):
    # load the result image and display it
    try:
        print("[RESULT] {result_num}. {result} - {score:.4f}".format(result_num=i + 1,
                                                                     result=resultID, score=score))

        result = cv2.imread("{}".format(im_paths[resultID]))
        montage.addResult(result, text="#{}".format(i + 1),
                          highlight=resultID in queryRelevant)
    except:
        print('Error: exception found on print')

# show the output image of results
cv2.imshow("Results", imutils.resize(montage.montage, height=700))

# save results
this_dir = osp.dirname(__file__)
out_results_DIR = os.path.join(this_dir,'results/')
if not os.path.exists(out_results_DIR):
    os.mkdir(out_results_DIR)

qry_pn = args["query"]
out_results_file = os.path.join(out_results_DIR, qry_pn[qry_pn.rfind("/") + 1:])
cv2.imwrite(out_results_file, imutils.resize(montage.montage, height=700))

cv2.waitKey(0)
dSearcher.finish()


# def binary_hash_codes(feature_mat):
# #     """convert feature matrix of latent layer to binary hash codes"""
# #     xs, ys = np.where(feature_mat > 0.5)
# #     code_mat = np.zeros(feature_mat.shape)
# #
# #     for i in range(len(xs)):
# #         code_mat[xs[i]][ys[i]] = 1
# #
# #     return code_mat
# #
# #
# # def retrieve_image(target_image, model_file, deploy_file, imagemean_file,
# #                    threshold=1):
# #     model_dir = os.path.dirname(model_file)
# #     image_files = np.load(os.path.join(model_dir, 'image_files.npy'))
# #     fc7_feature_mat = np.load(os.path.join(model_dir, 'fc7_features.npy'))
# #     latent_feature_file = os.path.join(model_dir, 'latent_features.npy')
# #     latent_feature_mat = np.load(latent_feature_file)
# #
# #     candidates = []
# #     dist = 0
# #     for layer, mat in layer_features(['latent', 'fc7'], model_file,
# #                                      deploy_file, imagemean_file,
# #                                      [target_image], show_pred=True):
# #         if layer == 'latent':
# #             # coarse-level search
# #             mat = binary_hash_codes(mat)
# #             mat = mat * np.ones((latent_feature_mat.shape[0], 1))
# #             dis_mat = np.abs(mat - latent_feature_mat)
# #             hamming_dis = np.sum(dis_mat, axis=1)
# #             distance_file = os.path.join(model_dir, 'hamming_dis.npy')
# #             np.save(distance_file, hamming_dis)
# #             candidates = np.where(hamming_dis < threshold)[0]
# #
# #         if layer == 'fc7':
# #             # fine-level search
# #             kdt = KDTree(fc7_feature_mat[candidates], metric='euclidean')
# #             k = 6
# #
# #             if not candidates.shape[0] > 6:
# #                 k = candidates.shape[0]
# #
# #             dist, idxs = kdt.query(mat, k=k)
# #             candidates = candidates[idxs]
# #             print(dist)
# #
# #     return image_files[candidates][0], dist[0]
# #
# #
# # if __name__ == '__main__':
# #     import sys
# #     if len(sys.argv) != 5:
# #         usage = 'Usage: python retrieve.py' + \
# #                 ' model_file deploy_file imagemean_file target_image.jpg'
# #         print(usage)
# #     else:
# #         model_file = sys.argv[1]
# #         deploy_file = sys.argv[2]
# #         imagemean_file = sys.argv[3]
# #         target_image = sys.argv[4]
# #
# #         is_exists = os.path.exists(model_file) and os.path.exists(deploy_file)\
# #             and os.path.exists(imagemean_file)
# #
# #         if is_exists:
# #             res, _ = retrieve_image(target_image, model_file, deploy_file,
# #                                     imagemean_file, threshold=5)
# #             print(res)
# #             if not os.path.exists('results'):
# #                 os.mkdir('results')
# #             for i in range(len(res)):
# #                 subprocess.call(['cp', res[i], 'results/%s.jpg' % str(i)])
# #         else:
# #             print('The model related files may not exit')
# #             print('Please check files: {}, {}, {}'
# #                   .format(model_file, deploy_file, imagemean_file))
