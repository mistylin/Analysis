from __future__ import division
import pandas as pd
#from math import *
import numpy as np
import os
import glob
import matplotlib.pylab as pl
from IPython import embed as shell
import seaborn as sn

sn.set_style('ticks')
data_dir = '/Users/xiaomeng/disks/Aeneas_Raw/2017/visual/Attention/Behavioural/'
sublist =  ['SL','MS']

def compute_behavioral_performance (csv_files):
	''' compute 
	1) RT, accuracy for each run, and averaged values over runs
	2) RT for different condiitons 
	(correct response, ,wrong response, wrong task, wrong direction)'''

	print ' RT  &  accuracy '
	reaction_time = []
	all_reponses = []
	task = []
	button =[]
	position_x =[]
	position_y =[]

	for this_file in csv_files: # loop over files

		csv_data = pd.read_csv(this_file)  #load data na_values = ["NaN"]
		
		# each run
		RT_run = np.nanmean(csv_data['reaction_time'])
		accuracy_run = np.nanmean(csv_data['correct_answer'])

		reaction_time.extend(csv_data['reaction_time'])
		all_reponses.extend(csv_data['correct_answer'])
		task.extend(csv_data['task'])
		button.extend(csv_data['button'])
		position_x.extend(csv_data['trial_position_x'])
		position_y.extend(csv_data['trial_position_y'])
		
		print RT_run, accuracy_run

	# average across runs
	RT_mean_sub = np.nanmean(np.array(reaction_time))
	accuracy_sub = np.nanmean(np.array(all_reponses))
	RT_correct = np.nanmean(np.array(reaction_time)[np.array(all_reponses) == 1])
	RT_wrong = np.nanmean(np.array(reaction_time)[np.array(all_reponses) == 0])

	wrong_task_mask = np.array(((np.array(task)==2)*((np.array(button)=='s')+(np.array(button)=='f')))+((np.array(task)==1)*((np.array(button)=='j')+(np.array(button)=='l'))))
	wrong_direction_mask = np.array((np.array(all_reponses)==0) * (~np.array(wrong_task_mask)))
	RT_wrong_task = np.nanmean(np.array(reaction_time)[wrong_task_mask])
	RT_wrong_direction = np.nanmean(np.array(reaction_time)[wrong_direction_mask])

	# four locations
	top_left_mask = np.array((np.array(position_x) == -2.5) * (np.array(position_y) == 2.5))
	top_right_mask = np.array((np.array(position_x) == 2.5) * (np.array(position_y) == 2.5))
	bottom_left_mask = np.array((np.array(position_x) == -2.5) * (np.array(position_y) == -2.5))
	bottom_right_mask = np.array((np.array(position_x) == 2.5) * (np.array(position_y) == -2.5))

	## top left (x = -2.5, y =2.5)
	RT_mean_sub_tl = np.nanmean(np.array(reaction_time)[top_left_mask])
	accuracy_sub_tl = np.nanmean(np.array(all_reponses)[top_left_mask])
	RT_correct_tl = np.nanmean(np.array(reaction_time)[(np.array(all_reponses) == 1)* top_left_mask])
	RT_wrong_tl = np.nanmean(np.array(reaction_time)[(np.array(all_reponses) == 0) * top_left_mask])

	## top right (x = 2.5, y =2.5)
	RT_mean_sub_tr = np.nanmean(np.array(reaction_time)[top_right_mask])
	accuracy_sub_tr = np.nanmean(np.array(all_reponses)[top_right_mask])
	RT_correct_tr = np.nanmean(np.array(reaction_time)[(np.array(all_reponses) == 1)* top_right_mask])
	RT_wrong_tr = np.nanmean(np.array(reaction_time)[(np.array(all_reponses) == 0) * top_right_mask])

	## bottom left (x = -2.5, y =-2.5)
	RT_mean_sub_bl = np.nanmean(np.array(reaction_time)[bottom_left_mask])
	accuracy_sub_bl = np.nanmean(np.array(all_reponses)[bottom_left_mask])
	RT_correct_bl = np.nanmean(np.array(reaction_time)[(np.array(all_reponses) == 1)* bottom_left_mask])
	RT_wrong_bl = np.nanmean(np.array(reaction_time)[(np.array(all_reponses) == 0) * bottom_left_mask])

	## bottom right (x = 2.5, y =-2.5)
	RT_mean_sub_br = np.nanmean(np.array(reaction_time)[bottom_right_mask])
	accuracy_sub_br = np.nanmean(np.array(all_reponses)[bottom_right_mask])
	RT_correct_br = np.nanmean(np.array(reaction_time)[(np.array(all_reponses) == 1)* bottom_right_mask])
	RT_wrong_br = np.nanmean(np.array(reaction_time)[(np.array(all_reponses) == 0) * bottom_right_mask])


	print 'average', RT_mean_sub, accuracy_sub
	print 'RT_correct', RT_correct
	print 'RT_wrong', RT_wrong 
	print 'RT_wrong_task', RT_wrong_task
	print 'RT_wrong_direction', RT_wrong_direction

	print 'top left: RT_mean: %s, accuracy: %s, RT_correct: %s, RT_wrong: %s' %( str(RT_mean_sub_tl), str(accuracy_sub_tl), str(RT_correct_tl), str(RT_wrong_tl))
	print 'top right: RT_mean: %s, accuracy: %s, RT_correct: %s, RT_wrong: %s' %( str(RT_mean_sub_tr), str(accuracy_sub_tr), str(RT_correct_tr), str(RT_wrong_tr))
	print 'bottom left: RT_mean: %s, accuracy: %s, RT_correct: %s, RT_wrong: %s' %( str(RT_mean_sub_bl), str(accuracy_sub_bl), str(RT_correct_bl), str(RT_wrong_bl))
	print 'bottom right: RT_mean: %s, accuracy: %s, RT_correct: %s, RT_wrong: %s' %( str(RT_mean_sub_br), str(accuracy_sub_br), str(RT_correct_br), str(RT_wrong_br))

	return np.array(all_reponses)


def cumulative_sum(x):

	return np.array([np.nansum(x[:i]) for i in range(len(x))]) #range(len(x))+1])


def plot_staircase(csv_files, subname):
	'''plot stairecase for each participant'''
	
	all_reponses = compute_behavioral_performance(csv_files)
	n_responses = np.arange(1, len(all_reponses)+1)
	#cum_responses = np.cumsum(all_reponses)
	cum_responses = cumulative_sum(all_reponses)
	staircase = cum_responses/n_responses
	
	f = pl.figure()
	#s = f.add_subplot(111)
	pl.plot(staircase)
	#s.set_title('staircase')
	pl.ylim([0, 1])
	pl.axhline(0.75,color='k',ls='--')
	sn.despine(offset=10)
	#s.set_title('staircase')
	#pl.show()
	pl.savefig('/Users/xiaomeng/disks/Aeneas_Home/Analysis/%s_staircase_plot.pdf'%(subname))

	# accuracy_step = []
	# staircase = []
	# for i in len(all_reponses):
	# 	cum_accuracy = np.nanmean(all_reponses[:i])
		
	# 	accuracy_step.extend(item)
	# 	addition = np.nanmean(accuracy_step)
	# 	staircase.extend(addition)
	# pl.plot(staircase)


# loop over the subjects
for subii, subname in enumerate(sublist):

	print '[main] Running analysis for No. %s participant %s' % (str(subii+1), subname)
	subject_dir= os.path.join(data_dir,subname)
	csv_files = glob.glob(subject_dir+'/*.csv')

	#compute_behavioral_performance(csv_files)
	plot_staircase(csv_files,subname)


	
# # not useful anymore
# 	runnum = str(subii+1)
# 	csv_filename = data_dir + subname + '/'+ subname.lower() +'_' + runnum + '_output.csv'
# 	csv_filename = "%s%s/%s_%d_output.csv" % (data_dir, subname, subname.lower() + runnum)











