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



standard_parameters = {
	
	## common parameters:
	'TR':               	 0.945,		# VERY IMPORTANT TO FILL IN!! (in secs)
	'mapper_ntrials':		      128,# trials per location
	'mapper_max_index': 63,

	# For custom eye tracker points
	'eyelink_calib_size': 0.5,
	'x_offset':			  0.0,
	
	## stimulus parameters:calc
	'stimulus_size': [2.5, 100],#2.5,#100,#2.5,#1.5,	# diameter in dva


	'stimulus_mask': ['raisedCos',None],#'raisedCos',#None,
	'stimulus_positions': [[[1.5, 1.5], [1.5, -1.5], [-1.5, -1.5], [-1.5, 1.5]], [[0.0,0.0]]],#(0.0, 0.0),

	'stimulus_base_spatfreq': [0.03,0.06],#4,#0.04,#0.02,#0.04,

	'stimulus_base_orientation': (0,90),#(45, 135),
	'stimulus_base_colors': ((55,80,75), (55,-80,75)),

	'quest_initial_stim_values': (70,70,5,5),# (50, 50, 5),

	'quest_stepsize': [15,15,2,2],
				 
	'quest_r_index': (0),#(0,1),
	'quest_g_index': (1),#(2,3),
	'quest_h_index': (2),#(4,5),
	'quest_v_index': (3),

	'quest_minmax': [(0,80),(0,80),(0,100),(0,100)],
	

	'session_types': [0,1,2,3],
	'tasks': [1,2],

	## timing of the presentation:

	'timing_start_empty': 0,#15,
	'timing_finish_empty': 0,#15,

	'timing_stim_1_Duration' : .15, # duration of stimulus presentation, in sec
	'timing_ISI'             : .03,
	'timing_stim_2_Duration' : .15, # duration of stimulus presentation, in sec
	'timing_cue_duration'    : 0.75,	# Duration for each cue separately
	'timing_stimcue_interval' : 0.5,
	'timing_responseDuration' : 1.5,#2.75, # time to respond	
	'timing_ITI_duration':  (0.5, 1.5),		# in sec

	'response_buttons_orientation': ['b','y'],#['j','l'], #
	'response_buttons_color': ['w','e'],#['s','f'],#

	# mapper location order (from above):
	# (T=top,B=bottom,L=left,R=right)
	# TR-BR-BL-TL

	# OriColorMapper stuff
	'stimulus_ori_min':			90.0,		# minimal orientation to show (in deg)
	'stimulus_ori_max':		   270.0,		# maximum orientation to show (in deg)
	'stimulus_ori_steps':		   8,		# how many steps between min and max
	'stimulus_col_min': 		   0,		# start point on color circle
	'stimulus_col_max':			   8,		# end point on color circle
	'stimulus_col_steps':		   8,		# how many steps through colorspace
	'stimulus_col_rad':			  75,		# radius of color circle
	'stimulus_col_baselum':		  55,	    # L

	'mapper_pre_post_trials':		 15,
	'mapper_stimulus_duration':      0.65,		# in TR
	'mapper_task_duration':			 0.5,    # in TR
	'mapper_response_duration':		 1.0,	 # in TR
	'mapper_task_timing':			(2.0, 8.0), # min and max separation of fix task
	'mapper_ITI_duration':           0.2,		# in TR
	'mapper_n_redraws':		 	 	 5.0,		# refresh random phase this many times during presentation	
	'mapper_mapper_redraws':		20.0	

}

def prepare_trials(standard_parameters):
	"""docstring for prepare_trials(self):"""

	standard_parameters = standard_parameters


	orientations = np.linspace(standard_parameters['stimulus_ori_min'], standard_parameters['stimulus_ori_max'], standard_parameters['stimulus_ori_steps']+1)[:-1]

	# Compute evenly-spaced steps in (L)ab-space

	color_theta = (np.pi*2)/standard_parameters['stimulus_col_steps']
	color_angle = color_theta * np.arange(standard_parameters['stimulus_col_min'], standard_parameters['stimulus_col_max'],dtype=float)
	color_radius = standard_parameters['stimulus_col_rad']

	color_a = color_radius * np.cos(color_angle)
	color_b = color_radius * np.sin(color_angle)

	# colors = [ct.lab2psycho((standard_parameters['stimulus_col_baselum'], a, b)) for a,b in zip(color_a, color_b)]			 
	colors = [(standard_parameters['stimulus_col_baselum'], a, b) for a,b in zip(color_a, color_b)]			 

	#stimulus_positions = standard_parameters['stimulus_positions']
	
	trial_array = []

	# self.trial_array = np.array([[[o,c[0],c[1],c[2]] for o in self.orientations] for c in self.colors]).reshape((self.standard_parameters['stimulus_ori_steps']*self.standard_parameters['stimulus_col_steps'],4))
	trial_array = np.array([[[o,c[0],c[1], c[2]] for o in orientations] for c in colors]).reshape((standard_parameters['stimulus_ori_steps']*standard_parameters['stimulus_col_steps'],4))
	
	return trial_array, colors

trial_array = prepare_trials(standard_parameters)[0]
colors = prepare_trials(standard_parameters)[1]

print trial_array
print trial_array.shape
print colors
# print colors.shape






