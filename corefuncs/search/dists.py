#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zhleternity
# -------------------------------------

import numpy as np

def chi2_distance(histA, histB, eps=1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum(((np.array(histA, dtype='float32') - np.array(histB,  dtype='float32')) ** 2)
                     / (np.array(histA, dtype='float32') + np.array(histB, dtype='float32') + eps))

    # return the chi-squared distance
    return d
