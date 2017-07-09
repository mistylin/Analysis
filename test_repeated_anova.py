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

################-----------------------------------------------------------
########### for 1 variable
################-----------------------------------------------------------

N = 5 # number of voxels: n_voxels
P = [1,2,3] #  64ch vs. 96ch
# Q = [1,2,3] # V1 vs. V4

### prepare a dataset
values = [[998], [1119], [1300]]  
mus = np.concatenate([np.repeat(value, N) for value in values]).tolist() # len: 120
rt = np.random.normal(mus, scale=112.0, size=N*len(P)).tolist()  
# rt = 20*998   20*511   20*1119  20*620   20*1300   20*790
#  64ch(V1) vs. 96ch(V1), 64ch(V4) vs. 96ch(V4)

# sub_id = 6* 1-20
# iv1 = 20*1, 20*2,  20*1, 20*2, 20*1, 20*2
# iv2 = 40*1, 40*2, 40*3


voxel_id = [i+1 for i in xrange(N)]*(len(P)) # 1-20 1-20 ... so 120 in total. 
iv1 = np.concatenate([np.array([p]*N) for p in P]).tolist() 

 
shell()
Sub = namedtuple('Sub', ['voxel_id', 'rt','iv1'])               
df = pt.DataFrame()
 
for idx in xrange(len(voxel_id)):
	df.insert(Sub(voxel_id[idx],rt[idx], iv1[idx])._asdict()) 


# box_plot
df.box_plot('rt', factors=['iv1'])


# Two-way ANOVA
aov = df.anova('rt', sub='Sub_id', wfactors=['iv1'])
print(aov)


################-----------------------------------------------------------
########### for 1 variable, specific for color ori mapping exp
################-----------------------------------------------------------
N = 5 # number of voxels: n_voxels
n_condition_iv1 = [1,2,3] #  64ch vs. 8ch vs. 8ch
r_squareds = np.random.normal(mus, scale=112.0, size=N*len(n_condition_iv1)).tolist()  
voxel_id = [i+1 for i in xrange(N)]*(len(n_condition_iv1)) # 1-20 1-20 ... so 120 in total. 
model_type = np.concatenate([np.array([p]*N) for p in n_condition_iv1]).tolist() 
 
# shell()
Sub = namedtuple('Sub', ['voxel_id', 'r_squareds','model_type'])               
df = pt.DataFrame()
for idx in xrange(len(voxel_id)):
	df.insert(Sub(voxel_id[idx],r_squareds[idx], model_type[idx])._asdict()) 

# box_plot
df.box_plot('r_squareds', factors=['model_type'])

# Two-way ANOVA
aov = df.anova('r_squareds', sub='voxel_id', wfactors=['model_type'])
print(aov)



################-----------------------------------------------------------
########### for 2 variables
################-----------------------------------------------------------
#### two variables
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




sub-n001
/home/xiaomeng/anaconda3/envs/py27/lib/python2.7/site-packages/pyvttbl/stats/_anova.py:1240: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
  return list(array(list(zeros((p-len(b))))+b)+1.)
r_squareds ~ model_type

TESTS OF WITHIN SUBJECTS EFFECTS

Measure: r_squareds
     Source                              Type III    eps       df        MS         F       Sig.   et2_G   Obs.      SE       95% CI    lambda     Obs.  
                                            SS                                                                                                     Power 
========================================================================================================================================================
model_type          Sphericity Assumed     97.143       -          2   48.571   11074.022      0   0.968   5228   9.160e-04    0.002   11076.140       1 
                    Greenhouse-Geisser     97.143   0.503      1.006   96.550   11074.022      0   0.968   5228   9.160e-04    0.002   11076.140       1 
                    Huynh-Feldt            97.143   0.503      1.006   96.550   11074.022      0   0.968   5228   9.160e-04    0.002   11076.140       1 
                    Box                    97.143   0.500          1   97.143   11074.022      0   0.968   5228   9.160e-04    0.002   11076.140       1 
--------------------------------------------------------------------------------------------------------------------------------------------------------
Error(model_type)   Sphericity Assumed     45.852       -      10454    0.004                                                                            
                    Greenhouse-Geisser     45.852   0.503   5259.100    0.009                                                                            
                    Huynh-Feldt            45.852   0.503   5259.100    0.009                                                                            
                    Box                    45.852   0.500       5227    0.009                                                                            

TABLES OF ESTIMATED MARGINAL MEANS

Estimated Marginal Means for model_type
model_type   Mean    Std. Error   95% Lower Bound   95% Upper Bound 
===================================================================
64ch         0.208        0.002             0.205             0.212 
8ch_col      0.039    3.608e-04             0.038             0.039 
8ch_ori      0.044    4.121e-04             0.043             0.045 






sub-n005
/home/xiaomeng/anaconda3/envs/py27/lib/python2.7/site-packages/pyvttbl/stats/_anova.py:1240: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
  return list(array(list(zeros((p-len(b))))+b)+1.)
r_squareds ~ model_type

TESTS OF WITHIN SUBJECTS EFFECTS

Measure: r_squareds
     Source                              Type III    eps       df        MS        F       Sig.   et2_G   Obs.      SE       95% CI    lambda    Obs.  
                                            SS                                                                                                   Power 
======================================================================================================================================================
model_type          Sphericity Assumed     21.458       -          2   10.729   6099.771      0   0.297   5171   5.833e-04    0.001   6100.950       1 
                    Greenhouse-Geisser     21.458   0.505      1.010   21.239   6099.771      0   0.297   5171   5.833e-04    0.001   6100.950       1 
                    Huynh-Feldt            21.458   0.505      1.010   21.239   6099.771      0   0.297   5171   5.833e-04    0.001   6100.950       1 
                    Box                    21.458   0.500          1   21.458   6099.771      0   0.297   5171   5.833e-04    0.001   6100.950       1 
------------------------------------------------------------------------------------------------------------------------------------------------------
Error(model_type)   Sphericity Assumed     18.187       -      10340    0.002                                                                          
                    Greenhouse-Geisser     18.187   0.505   5223.332    0.003                                                                          
                    Huynh-Feldt            18.187   0.505   5223.332    0.003                                                                          
                    Box                    18.187   0.500       5170    0.004                                                                          

TABLES OF ESTIMATED MARGINAL MEANS

Estimated Marginal Means for model_type
model_type   Mean    Std. Error   95% Lower Bound   95% Upper Bound 
===================================================================
64ch         0.137        0.001             0.135             0.140 
8ch_col      0.060    6.492e-04             0.058             0.061 
8ch_ori      0.058    6.464e-04             0.056             0.059 


In [4]: t_64vs8ori_r2_rel, p_64vs8ori_r2_rel
Out[4]: (78.548333850736128, 0.0)

In [5]: t_64vs8col_r2_rel, p_64vs8col_r2_rel
Out[5]: (78.224199180644419, 0.0)

In [6]: t_8oriVs8col_r2_rel, p_8oriVs8col_r2_rel
Out[6]: (-16.385958672186558, 7.1847028723497315e-59)























