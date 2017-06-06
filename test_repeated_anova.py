# from __future__ import division
import os,sys,glob

import numpy as np
import scipy.stats as stats
from scipy.signal import fftconvolve
import nibabel as nib
import pickle

from Staircase import ThreeUpOneDownStaircase
from tools import two_gamma as hrf

from IPython import embed as shell
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.gridspec import GridSpec
from scipy import ndimage

from scipy.stats.stats import pearsonr
import pandas as pd

from sklearn.linear_model import RidgeCV


import numpy as np
import pyvttbl as pt
from collections import namedtuple



N = 20 # number of voxels: n_voxels
P = [1,2] #  64ch vs. 96ch
Q = [1,2,3] # V1 vs. V4

values = [[998,511], [1119,620], [1300,790]]    rt = 20*998   20*511   20*1119  20*620   20*1300   20*790
#  64ch(V1) vs. 96ch(V1), 64ch(V4) vs. 96ch(V4)

sub_id = 6* 1-20
iv1 = 20*1, 20*2,  20*1, 20*2, 20*1, 20*2
iv2 = 40*1, 40*2, 40*3


sub_id = [i+1 for i in xrange(N)]*(len(P)*len(Q)) # 1-20 1-20 ... so 120 in total. 
mus = np.concatenate([np.repeat(value, N) for value in values]).tolist() # len: 120
rt = np.random.normal(mus, scale=112.0, size=N*len(P)*len(Q)).tolist()



iv1 = np.concatenate([np.array([p]*N) for p in P]*len(Q)).tolist() 
iv2 = np.concatenate([np.array([q]*(N*len(P))) for q in Q]).tolist()
 
shell()
Sub = namedtuple('Sub', ['Sub_id', 'rt','iv1', 'iv2'])               
df = pt.DataFrame()
 
for idx in xrange(len(sub_id)):
	df.insert(Sub(sub_id[idx],rt[idx], iv1[idx],iv2[idx])._asdict()) 


# box_plot
df.box_plot('rt', factors=['iv1', 'iv2'])


# Two-way ANOVA
aov = df.anova('rt', sub='Sub_id', wfactors=['iv1', 'iv2'])
print(aov)


