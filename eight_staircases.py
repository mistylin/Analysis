from __future__ import division
import pandas as pd
#from math import *
import numpy as np
import scipy
import os
import glob
import matplotlib.pylab as pl
from IPython import embed as shell
import seaborn as sn
import sys
from scipy.stats.stats import pearsonr
import pylab
from psychopy import data #, gui, core
from psychopy.tools.filetools import fromFile

sn.set_style('ticks')

run_on_aeneas = True

# if run_on_aeneas:
# 	data_dir = '/home/raw_data/2017/visual/Attention/Behavioural/Pre_scan_data/'
# 	figure_dir = '/home/shared/2017/visual/Attention/behaviour/'
# else:
# 	data_dir = '/Users/xiaomeng/disks/Aeneas_Raw/2017/visual/Attention/Behavioural/Pre_scan_data/'
# 	figure_dir = '/Users/xiaomeng/disks/Aeneas_Shared/2017/visual/Attention/behaviour/'

if run_on_aeneas:
	data_dir = '/home/xiaomeng/Data/Pre_scan_data/'
	# figure_dir = '/home/shared/2017/visual/Attention/behaviour/'
	figure_dir = '/home/xiaomeng/Data/Pre_scan_data/'
else:
	data_dir = '/Users/xiaomeng/disks/Aeneas_Home/Data/Pre_scan_data/'
	figure_dir = '/Users/xiaomeng/disks/Aeneas_Shared/2017/visual/Attention/behaviour/'

'''fMRI beh >> 1)change sublist, 2)csv_files path, 3)buttons 4) savefig folders 5) locaitons 6) pop out csv_files with 0 7)'''

#data_dir = '/Users/xiaomeng/disks/Aeneas_Raw/2017/visual/Attention/Behavioural/'
sublist = ["xy"] #[ 'az', 'da', 'fh', 'hf', 'im', 'pl', 'rr', 'xy', 'mw', 'mb', 'as','vk']  # 'mw' #'SL','MS'- their staircases are not spearated, have problems when converting into graded_color
#['xy']
def load_beh_data(csv_files):
	'''extend data over runs, print RT, accuracy for each run'''
	#print ' RT  &  accuracy '
	reaction_time = []
	all_responses = []
	task = []
	button =[]
	position_x =[]
	position_y =[]
	trial_color  = []
	trial_ori = []
	trial_stimulus =[]

	#to calculate d prime
	response = []
	trial_direction = []

	# shell()
	for this_file in csv_files: # loop over files

		#shell()
		csv_data = pd.read_csv(this_file)  #load data na_values = ["NaN"]
		

		reaction_time.extend(csv_data['reaction_time'])
		all_responses.extend(csv_data['correct_answer'])
		task.extend(csv_data['task'])
		button.extend(csv_data['button'])
		position_x.extend(csv_data['trial_position_x'])
		position_y.extend(csv_data['trial_position_y'])
		trial_color.extend(csv_data['trial_color'])
		trial_ori.extend(csv_data['trial_orientation'])
		trial_stimulus.extend(csv_data['trial_stimulus'])
		response.extend(csv_data['response'])
		trial_direction.extend(['trial_direction'])
	
	return np.array(reaction_time), np.array(all_responses), np.array(task), np.array(button), np.array(position_x), np.array(position_y), np.array(trial_color), np.array(trial_ori), np.array(trial_stimulus), np.array(response), np.array(trial_direction)


def create_masks():
	'''create masks'''
	reaction_time, all_responses, task, button, position_x, position_y, trial_color, trial_ori, trial_stimulus, response, trial_direction = load_beh_data(csv_files) #  all the data

	
	color_task_mask = np.array(np.array(task)==1)
	ori_task_mask = np.array(np.array(task)==2)

	red_task_mask = np.array(color_task_mask * ((trial_stimulus == 0) + (trial_stimulus == 1)))
	gre_task_mask = np.array(color_task_mask * ((trial_stimulus == 2) + (trial_stimulus == 3)))
	hor_task_mask = np.array(ori_task_mask * ((trial_stimulus == 0) + (trial_stimulus == 2)))
	ver_task_mask = np.array(ori_task_mask * ((trial_stimulus == 1) + (trial_stimulus == 3)))
	
	# right_task_masks with nan values
	right_task_mask = np.array(((np.array(task)==1)*((np.array(button)=='s')+(np.array(button)=='f')))+((np.array(task)==2)*((np.array(button)=='j')+(np.array(button)=='l')))) # sfjl
	wrong_task_mask = np.array(((np.array(task)==2)*((np.array(button)=='s')+(np.array(button)=='f')))+((np.array(task)==1)*((np.array(button)=='j')+(np.array(button)=='l')))) # sfjl

	# responses_mask(correct+incorrect): delete nanvalues and invalid values(wrong task)!!
	responses_mask = np.array((np.array(all_responses)==1) + ((np.array(all_responses)==0) * right_task_mask)) 

	correct_answer_mask = np.array(np.array(all_responses)==1) 
	incorrect_answer_mask = np.array((np.array(all_responses)==0) * (~np.array(wrong_task_mask)))

	# four locations
	top_left_mask = np.array((np.array(position_x) == -2.5) * (np.array(position_y) == 2.5))
	top_right_mask = np.array((np.array(position_x) == 2.5) * (np.array(position_y) == 2.5))
	bottom_left_mask = np.array((np.array(position_x) == -2.5) * (np.array(position_y) == -2.5))
	bottom_right_mask = np.array((np.array(position_x) == 2.5) * (np.array(position_y) == -2.5))

	# compared with red_task_mask, these masks gives more trials (red stimulus, regardless the tasks -color or orientation)
	red_stimulus_mask = np.array((trial_stimulus == 0) + (trial_stimulus == 1))
	gre_stimulus_mask = np.array((trial_stimulus == 2) + (trial_stimulus == 3))
	hor_stimulus_mask = np.array((trial_stimulus == 0) + (trial_stimulus == 2))
	ver_stimulus_mask = np.array((trial_stimulus == 1) + (trial_stimulus == 3))

	return color_task_mask, ori_task_mask, red_task_mask, gre_task_mask, hor_task_mask, ver_task_mask, right_task_mask, wrong_task_mask, responses_mask, correct_answer_mask, incorrect_answer_mask, top_left_mask, top_right_mask, bottom_left_mask, bottom_right_mask, red_stimulus_mask, gre_stimulus_mask, hor_stimulus_mask, ver_stimulus_mask


def plot_staircase(csv_files, subname):
	'''plot stairecase for each participant'''
	
	all_responses = load_beh_data(csv_files)[1] # index starts from 0
	responses_mask = create_masks()[8] # changes! if add seperatable analysis

	red_task_mask = create_masks()[2]
	gre_task_mask = create_masks()[3]
	hor_task_mask = create_masks()[4]
	ver_task_mask = create_masks()[5]

	# color & orientation accuracy

	red_task_responses = np.array(all_responses)[responses_mask * red_task_mask]
	red_task_n_responses = np.arange(1, len(red_task_responses)+1)
	red_task_cum_responses = np.cumsum(red_task_responses)
	red_task_accuracy = red_task_cum_responses/red_task_n_responses
	
	gre_task_responses = np.array(all_responses)[responses_mask * gre_task_mask]
	gre_task_n_responses = np.arange(1, len(gre_task_responses)+1)
	gre_task_cum_responses = np.cumsum(gre_task_responses)
	gre_task_accuracy = gre_task_cum_responses/gre_task_n_responses

	hor_task_responses = np.array(all_responses)[responses_mask * hor_task_mask]
	hor_task_n_responses = np.arange(1, len(hor_task_responses)+1)
	hor_task_cum_responses = np.cumsum(hor_task_responses)
	hor_task_accuracy = hor_task_cum_responses/hor_task_n_responses

	ver_task_responses = np.array(all_responses)[responses_mask * ver_task_mask]
	ver_task_n_responses = np.arange(1, len(ver_task_responses)+1)
	ver_task_cum_responses = np.cumsum(ver_task_responses)
	ver_task_accuracy = ver_task_cum_responses/ver_task_n_responses

	# staircase
	trial_color = load_beh_data(csv_files)[6]
	trial_ori = load_beh_data(csv_files)[7]

	trial_index_mask = np.zeros((len(trial_color),))
	trial_index_mask[::2] = 1

	red_staircase_0 = np.abs(trial_color[responses_mask * red_task_mask * (trial_index_mask ==0)])
	red_staircase_1 = np.abs(trial_color[responses_mask * red_task_mask * (trial_index_mask ==1)])

	gre_staircase_0 = np.abs(trial_color[responses_mask * gre_task_mask * (trial_index_mask ==0)])
	gre_staircase_1 = np.abs(trial_color[responses_mask * gre_task_mask * (trial_index_mask ==1)])

	hor_staircase_0 = np.abs(trial_ori[responses_mask * hor_task_mask * (trial_index_mask ==0)])
	hor_staircase_1 = np.abs(trial_ori[responses_mask * hor_task_mask * (trial_index_mask ==1)])

	ver_staircase_0 = np.abs(trial_ori[responses_mask * ver_task_mask * (trial_index_mask ==0)])
	ver_staircase_1 = np.abs(trial_ori[responses_mask * ver_task_mask * (trial_index_mask ==1)])

	staircases = { 'red': [red_staircase_0, red_staircase_1],
				'green': [gre_staircase_0, gre_staircase_1],
				'horizontal':[hor_staircase_0, hor_staircase_1],
				'vertical': [ver_staircase_0, ver_staircase_1]
	}

	conditions = ['red', 'green','horizontal', 'vertical']
	#plot accuracy & staircase
	f = pl.figure(figsize = (25,15))

	for i in range(len(staircases)):
		s = f.add_subplot(2,4,i+1)
		pl.plot(staircases[conditions[i]][0])
		pl.plot(staircases[conditions[i]][1])
		sn.despine(offset=10)
		s.set_title('staircase ' + conditions[i], fontsize = 20)

	pl.show()

	s1 = f.add_subplot(241)

	pl.plot(red_task_accuracy)
	pl.plot(gre_task_accuracy)
	pl.legend(['red', 'green'], loc ='best', fontsize = 18)
	pl.ylim([0, 1])
	pl.axhline(0.79,color='k',ls='--')
	pl.axhline(0.71,color='k',ls='--')
	sn.despine(offset=10)
	s1.set_title('Moving color accuracy', fontsize = 20)

	s2 = f.add_subplot(232)
	#pl.plot(ori_accuracy)
	pl.plot(hor_task_accuracy)
	pl.plot(ver_task_accuracy)
	pl.legend(['horizontal', 'vertical'], loc ='best', fontsize = 18)
	pl.ylim([0, 1])
	pl.axhline(0.79,color='k',ls='--')
	pl.axhline(0.71,color='k',ls='--')
	sn.despine(offset=10)
	s2.set_title('Moving orientation accuracy', fontsize = 20)
	#s1.set_yaxis

	pl.savefig(figure_dir + 'lab_%s_color_ori_staircase_plot.jpg'%(subname))


# loop over the subjects
for subii, subname in enumerate(sublist):

	#print '[main] Running analysis for No. %s participant %s' % (str(subii+1), subname)
	subject_dir= os.path.join(data_dir,subname)
	csv_files = glob.glob(subject_dir+'/*.csv')
	csv_files.sort()

	# if csv_files[0].split('_')[2]=='0':
	# 	csv_files.pop(0)
	plot_staircase(csv_files,subname)





# def plot_staircases(initials,run_nr):


# 	stairs = ['red','green','ori']

# 	# Load staircase data
# 	staircases = pickle.load(open('data/' + initials + '_staircase.pickle','rb'))

# 	# shell()
# 	# Compute average performance over time
# 	percent_correct = list()
# 	for ii in range(len(staircases)):

# 		responses = staircases[ii].past_answers

# 		percent_correct.append(np.cumsum(responses) / np.arange(1,len(responses)+1))


# 	# Plot average resp correct over time

# 	f = pl.figure()

# 	for s in range(len(stairs)):
# 		pl.plot(percent_correct[s],'-')
# 	pl.legend(stairs)

# 	pl.savefig('data/%s_%d_staircase_plot.pdf'%(initials,run_nr))














