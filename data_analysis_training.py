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
	figure_dir = '/home/shared/2017/visual/Attention/behaviour/'
else:
	data_dir = '/Users/xiaomeng/disks/Aeneas_Home/Data/Pre_scan_data/'
	figure_dir = '/Users/xiaomeng/disks/Aeneas_Shared/2017/visual/Attention/behaviour/'

'''fMRI beh >> 1)change sublist, 2)csv_files path, 3)buttons 4) savefig folders 5) locaitons 6) pop out csv_files with 0 7)'''

#data_dir = '/Users/xiaomeng/disks/Aeneas_Raw/2017/visual/Attention/Behavioural/'
sublist =  ['SL','MS']

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

	# shell()
	for this_file in csv_files: # loop over files

		#shell()
		csv_data = pd.read_csv(this_file)  #load data na_values = ["NaN"]
		
		# # each run
		# RT_run = np.nanmean(csv_data['reaction_time'])
		# accuracy_run = np.nanmean(csv_data['correct_answer'])

		reaction_time.extend(csv_data['reaction_time'])
		all_responses.extend(csv_data['correct_answer'])
		task.extend(csv_data['task'])
		button.extend(csv_data['button'])
		position_x.extend(csv_data['trial_position_x'])
		position_y.extend(csv_data['trial_position_y'])
		trial_color.extend(csv_data['trial_color'])
		trial_ori.extend(csv_data['trial_orientation'])
		trial_stimulus.extend(csv_data['trial_stimulus'])

		# print RT_run, accuracy_run

	return np.array(reaction_time), np.array(all_responses), np.array(task), np.array(button), np.array(position_x), np.array(position_y), np.array(trial_color), np.array(trial_ori), np.array(trial_stimulus)


def create_masks():
	'''create masks'''
	reaction_time, all_responses, task, button, position_x, position_y, trial_color, trial_ori, trial_stimulus = load_beh_data(csv_files) #  all the data

	
	color_task_mask = np.array(np.array(task)==1)
	ori_task_mask = np.array(np.array(task)==2)

	red_task_mask = np.array(color_task_mask * ((trial_stimulus == 0) + (trial_stimulus == 1)))
	gre_task_mask = np.array(color_task_mask * ((trial_stimulus == 2) + (trial_stimulus == 3)))
	hor_task_mask = np.array(ori_task_mask * ((trial_stimulus == 0) + (trial_stimulus == 2)))
	ver_task_mask = np.array(ori_task_mask * ((trial_stimulus == 1) + (trial_stimulus == 3)))
	
	right_task_mask = np.array(((np.array(task)==1)*((np.array(button)=='s')+(np.array(button)=='f')))+((np.array(task)==2)*((np.array(button)=='j')+(np.array(button)=='l')))) # sfjl
	wrong_task_mask = np.array(((np.array(task)==2)*((np.array(button)=='s')+(np.array(button)=='f')))+((np.array(task)==1)*((np.array(button)=='j')+(np.array(button)=='l')))) # sfjl

	# responses_mask(correct+incorrect): delete nanvalues and invalid values(wrong task)!!
	responses_mask = np.array((np.array(all_responses)==1) + ((np.array(all_responses)==0)* right_task_mask)) 

	correct_answer_mask = np.array(np.array(all_responses)==1) 
	incorrect_answer_mask = np.array((np.array(all_responses)==0) * (~np.array(wrong_task_mask)))

	 #four locations
	top_left_mask = np.array((np.array(position_x) == -2.5) * (np.array(position_y) == 2.5))
	top_right_mask = np.array((np.array(position_x) == 2.5) * (np.array(position_y) == 2.5))
	bottom_left_mask = np.array((np.array(position_x) == -2.5) * (np.array(position_y) == -2.5))
	bottom_right_mask = np.array((np.array(position_x) == 2.5) * (np.array(position_y) == -2.5))

	return color_task_mask, ori_task_mask, red_task_mask, gre_task_mask, hor_task_mask, ver_task_mask, right_task_mask, wrong_task_mask, responses_mask, correct_answer_mask, incorrect_answer_mask, top_left_mask, top_right_mask, bottom_left_mask, bottom_right_mask 


def plot_staircase(csv_files, subname):
	'''plot stairecase for each participant'''
	
	all_responses = load_beh_data(csv_files)[1] # index starts from 0
	responses_mask = create_masks()[8] # changes! if add seperatable analysis
	correct_answer_mask = create_masks()[9]
	color_task_mask = create_masks()[0]
	ori_task_mask = create_masks()[1]
	red_task_mask = create_masks()[2]
	gre_task_mask = create_masks()[3]
	hor_task_mask = create_masks()[4]
	ver_task_mask = create_masks()[5]


	# color & orientation accuracy
	#responses = np.array(all_responses)[responses_mask]
	color_responses = np.array(all_responses)[responses_mask * color_task_mask]
	color_n_responses = np.arange(1, len(color_responses)+1)
	color_cum_responses = np.cumsum(color_responses)
	color_accuracy = color_cum_responses/color_n_responses

	ori_responses = np.array(all_responses)[responses_mask * ori_task_mask]
	ori_n_responses = np.arange(1, len(ori_responses)+1)
	ori_cum_responses = np.cumsum(ori_responses)
	ori_accuracy = ori_cum_responses/ori_n_responses

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

	red_staircase = np.abs(trial_color[correct_answer_mask * red_task_mask])
	gre_staircase = np.abs(trial_color[correct_answer_mask * gre_task_mask])
	hor_staircase = np.abs(trial_ori[correct_answer_mask * hor_task_mask])
	ver_staircase = np.abs(trial_ori[correct_answer_mask * ver_task_mask])

	# red_staircase = np.abs(trial_color[responses_mask * red_task_mask])
	# gre_staircase = np.abs(trial_color[responses_mask * gre_task_mask])
	# hor_staircase = np.abs(trial_ori[responses_mask * hor_task_mask])
	# ver_staircase = np.abs(trial_ori[responses_mask * ver_task_mask])

	#plot accuracy & staircase
	f = pl.figure(figsize = (25,15))

	s1 = f.add_subplot(231)
	pl.plot(color_accuracy)
	pl.plot(red_task_accuracy)
	pl.plot(gre_task_accuracy)
	pl.legend(['color', 'red', 'green'], loc ='best', fontsize = 18)
	pl.ylim([0, 1])
	pl.axhline(0.79,color='k',ls='--')
	sn.despine(offset=10)
	s1.set_title('Moving color accuracy', fontsize = 20)

	s2 = f.add_subplot(232)
	pl.plot(ori_accuracy)
	pl.plot(hor_task_accuracy)
	pl.plot(ver_task_accuracy)
	pl.legend(['orientation', 'horizontal', 'vertical'], loc ='best', fontsize = 18)
	pl.ylim([0, 1])
	pl.axhline(0.79,color='k',ls='--')
	sn.despine(offset=10)
	s2.set_title('Moving orientation accuracy', fontsize = 20)
	#s1.set_yaxis

	s3 = f.add_subplot(233)

	objects = ('red','green', 'horzontal', 'vertical')
	y_pos = np.arange(len(objects))
	y_values = [np.mean(red_task_accuracy[51:]), np.mean(gre_task_accuracy[51:]), np.mean(hor_task_accuracy[51:]), np.mean(ver_task_accuracy[51:])]
	sd = np.array([np.std(red_task_accuracy[51:]), np.std(gre_task_accuracy[51:]), np.std(hor_task_accuracy[51:]), np.std(ver_task_accuracy[51:])])
	n = np.array([np.array(red_task_accuracy[51:]).shape[0], np.array(gre_task_accuracy[51:]).shape[0], np.array(hor_task_accuracy[51:]).shape[0], np.array(ver_task_accuracy[51:]).shape[0] ])
	yerr = (sd/np.sqrt(n.squeeze()))*1.96
	# why shape? ValueError: In safezip, len(args[0])=4 but len(args[1])=1, !!! could use len()

	pl.bar(y_pos, y_values, yerr = yerr, align = 'center', alpha = 0.5)

	#sn.barplot( data = y_values, ci = 95, capsize = .2)
	#pl.errorbar(y_pos, y_values, yerr = yerr, fmt='-o')

	pl.axhline(0.79,color='k',ls='--')
	pl.xticks (y_pos, objects, fontsize = 40) # why doesn't work?

	#pl.ylable('percentage of correct')
	pl.title( 'accuracy for four conditions', fontsize = 20)
	pl.ylim([0, 1])
	sn.despine(offset=10)
	#s3.set_title('staircase color')

	s4 = f.add_subplot(234)
	pl.plot(red_staircase)
	pl.plot(gre_staircase)
	pl.legend(['red', 'green'], loc ='best', fontsize = 18)
	#pl.ylim([0, 50])
	sn.despine(offset=10)
	s4.set_title('staircase color', fontsize = 20)

	s5 = f.add_subplot(235)
	pl.plot(hor_staircase)
	pl.plot(ver_staircase)
	pl.legend(['horizontal', 'vertical'], loc ='best', fontsize = 18)
	#pl.ylim([0, 5])
	sn.despine(offset=10)
	s5.set_title('staircase orientation', fontsize = 20)

	#pl.show()
	pl.savefig(figure_dir + 'lab_%s_color_ori_staircase_plot.jpg'%(subname))
	#pl.savefig('/Users/xiaomeng/disks/Aeneas_Home/pdfs/%s_color_ori_staircase_plot.pdf'%(subname))


def compute_behavioral_performance(csv_files):
	''' compute 
	1) RT, accuracy for each run, and averaged values over runs
	2) RT for different condiitons 
	(correct response, ,wrong response, wrong task, wrong direction)'''


	reaction_time, all_responses, task, button, position_x, position_y, trial_color, trial_ori, trial_stimulus = load_beh_data(csv_files)

	color_task_mask, ori_task_mask, red_task_mask, gre_task_mask, hor_task_mask, ver_task_mask, right_task_mask, wrong_task_mask, responses_mask, correct_answer_mask, incorrect_answer_mask, top_left_mask, top_right_mask, bottom_left_mask, bottom_right_mask = create_masks()

	RT_color = np.array(reaction_time)[color_task_mask* (~np.isnan(reaction_time))] # without nan values
	RT_ori = np.array(reaction_time)[ori_task_mask* (~np.isnan(reaction_time))] # without nan values
	Accuracy_color = np.array(all_responses)[color_task_mask* (~np.isnan(reaction_time))] # without nan values
	Accuracy_ori = np.array(all_responses)[ori_task_mask* (~np.isnan(reaction_time))] # without nan values

	RT_correct = np.array(reaction_time)[correct_answer_mask* (~np.isnan(reaction_time))]
	RT_incorrect = np.array(reaction_time)[incorrect_answer_mask* (~np.isnan(reaction_time))]
	RT_right_task = np.array(reaction_time)[right_task_mask* (~np.isnan(reaction_time))]
	RT_wrong_task = np.array(reaction_time)[wrong_task_mask* (~np.isnan(reaction_time))]

	# GLM to test RT vs TASKVALUE or DISTRACTOR or interaction
	norm_color = np.abs(np.array(trial_color)) / np.abs(np.array(trial_color)).max()
	norm_ori = np.abs(np.array(trial_ori)) / np.abs(np.array(trial_ori)).max()

	TASKVALUE = np.zeros((len(reaction_time),))
	TASKVALUE[np.array(task)==1] = norm_color[np.array(task)==1]
	TASKVALUE[np.array(task)==2] = norm_ori[np.array(task)==2]
	TASKVALUE = TASKVALUE[correct_answer_mask* (~np.isnan(reaction_time))] #* ~np.isnan(all_responses)] # why use two creteria?

	DISTRACTOR = np.zeros((len(reaction_time),))
	DISTRACTOR[np.array(task)==2] = norm_color[np.array(task)==1] #how do you define the distractor?
	DISTRACTOR[np.array(task)==1] = norm_ori[np.array(task)==2]
	DISTRACTOR = DISTRACTOR[correct_answer_mask* (~np.isnan(reaction_time))]# * ~np.isnan(all_responses)]

	# select correct RTs, use log, things to be done>> use d' to replace design matrix!!
	RT_correct_log = np.log10(RT_correct)
	X = np.hstack([np.ones((len(RT_correct_log),1)), TASKVALUE[:,np.newaxis], DISTRACTOR[:,np.newaxis], TASKVALUE[:,np.newaxis]*DISTRACTOR[:,np.newaxis]])
	betas = np.linalg.lstsq(X, RT_correct_log)[0]
	#betas_new = np.linalg.pinv(X).dot(RT)
	SE = np.sqrt(np.sum((X.dot(betas) - RT_correct_log)**2)/(RT_correct_log.size - X.shape[1]))
	df= RT_correct_log.size - X.shape[1]
	t = [betas.squeeze().dot(contrast) / SE for contrast in np.array([[0,1,0,0],[0,0,1,0], [0,0,0,1]])]
	p = scipy.stats.t.sf(np.abs(t), df)*2

	# shell()
	# X1 = np.hstack([np.ones((len(RT),1)), TASKVALUE[:,np.newaxis]])
	# betas1 = np.linalg.lstsq(X1, RT)[0]
	# #betas_new = np.linalg.pinv(X).dot(RT)
	# SE1 = np.sqrt(np.sum((X1.dot(betas1) - RT)**2)/(RT.size - X1.shape[1]))
	# df1= RT.size - X1.shape[1]
	# t1 = [betas1.squeeze().dot(contrast1) / SE1 for contrast1 in np.array([0,1])]
	# p1 = scipy.stats.t.sf(np.abs(t1), df1)*2

	# Color vs Orientation on RT, %correct
	t_Col_vs_Ori_RT, p_Col_vs_Ori_RT = scipy.stats.ttest_ind(RT_color, RT_ori, equal_var=False, nan_policy='omit')
	t_Col_vs_Ori_Acc, p_Col_vs_Ori_Acc = scipy.stats.ttest_ind(Accuracy_color, Accuracy_ori, equal_var=False, nan_policy='omit')
	
	# correct vs incorrect answers on RT
	t_cor_vs_incor_RT, p_cor_vs_incor_RT = scipy.stats.ttest_ind(RT_correct, RT_incorrect, equal_var=False, nan_policy='omit')

	# right vs wrong tasks on RT
	t_righ_vs_wro_task_RT, p_righ_vs_wro_task_RT = scipy.stats.ttest_ind(RT_right_task, RT_wrong_task, equal_var=False, nan_policy='omit')

	# # Correct vs Incorrect task vs Incorrect direction on RT
	# F_answers_RT, p_answers_RT = scipy.stats.f_oneway(RT_correct, RT_wrong_task, RT_wrong_direction)
	# cor_vs_task = scipy.stats.ttest_ind(RT_correct, RT_wrong_task, equal_var=False, nan_policy='omit')
	# cor_vs_dir= scipy.stats.ttest_ind(RT_correct, RT_wrong_direction, equal_var=False, nan_policy='omit')
	# task_vs_dir = scipy.stats.ttest_ind(RT_wrong_task, RT_wrong_direction, equal_var=False, nan_policy='omit')

	return betas, t, p, t_Col_vs_Ori_RT, p_Col_vs_Ori_RT, t_Col_vs_Ori_Acc, p_Col_vs_Ori_Acc, t_cor_vs_incor_RT, p_cor_vs_incor_RT, t_righ_vs_wro_task_RT, p_righ_vs_wro_task_RT, RT_color, RT_ori, Accuracy_color, Accuracy_ori, RT_correct, RT_incorrect, RT_right_task, RT_wrong_task 

# 	# average across runs should changes names!!! adding 'means'!!!
# 	RT_mean_sub = np.nanmean(np.array(reaction_time))
# 	accuracy_sub = np.nanmean(np.array(all_responses))
# 	RT_correct = np.nanmean(np.array(reaction_time)[correct_answer_mask])
# 	RT_wrong = np.nanmean(np.array(reaction_time)[wrong_answer_mask])

# 	RT_wrong_task = np.nanmean(np.array(reaction_time)[wrong_task_mask])
# 	RT_wrong_direction = np.nanmean(np.array(reaction_time)[wrong_direction_mask])

# 	# four locations
# 	## top left (x = -2.5, y =2.5)
# 	RT_mean_sub_tl = np.nanmean(np.array(reaction_time)[top_left_mask])
# 	accuracy_sub_tl = np.nanmean(np.array(all_responses)[top_left_mask])
# 	RT_correct_tl = np.nanmean(np.array(reaction_time)[(np.array(all_responses) == 1)* top_left_mask])
# 	RT_wrong_tl = np.nanmean(np.array(reaction_time)[(np.array(all_responses) == 0) * top_left_mask])

# 	## top right (x = 2.5, y =2.5)
# 	RT_mean_sub_tr = np.nanmean(np.array(reaction_time)[top_right_mask])
# 	accuracy_sub_tr = np.nanmean(np.array(all_responses)[top_right_mask])
# 	RT_correct_tr = np.nanmean(np.array(reaction_time)[(np.array(all_responses) == 1)* top_right_mask])
# 	RT_wrong_tr = np.nanmean(np.array(reaction_time)[(np.array(all_responses) == 0) * top_right_mask])

# 	## bottom left (x = -2.5, y =-2.5)
# 	RT_mean_sub_bl = np.nanmean(np.array(reaction_time)[bottom_left_mask])
# 	accuracy_sub_bl = np.nanmean(np.array(all_responses)[bottom_left_mask])
# 	RT_correct_bl = np.nanmean(np.array(reaction_time)[(np.array(all_responses) == 1)* bottom_left_mask])
# 	RT_wrong_bl = np.nanmean(np.array(reaction_time)[(np.array(all_responses) == 0) * bottom_left_mask])

# 	## bottom right (x = 2.5, y =-2.5)
# 	RT_mean_sub_br = np.nanmean(np.array(reaction_time)[bottom_right_mask])
# 	accuracy_sub_br = np.nanmean(np.array(all_responses)[bottom_right_mask])
# 	RT_correct_br = np.nanmean(np.array(reaction_time)[(np.array(all_responses) == 1)* bottom_right_mask])
# 	RT_wrong_br = np.nanmean(np.array(reaction_time)[(np.array(all_responses) == 0) * bottom_right_mask])


# 	print 'average', RT_mean_sub, accuracy_sub
# 	print 'RT_correct', RT_correct
# 	print 'RT_wrong', RT_wrong 
# 	print 'RT_wrong_task', RT_wrong_task
# 	print 'RT_wrong_direction', RT_wrong_direction

# 	print 'top left: RT_mean: %s, accuracy: %s, RT_correct: %s, RT_wrong: %s' %( str(RT_mean_sub_tl), str(accuracy_sub_tl), str(RT_correct_tl), str(RT_wrong_tl))
# 	print 'top right: RT_mean: %s, accuracy: %s, RT_correct: %s, RT_wrong: %s' %( str(RT_mean_sub_tr), str(accuracy_sub_tr), str(RT_correct_tr), str(RT_wrong_tr))
# 	print 'bottom left: RT_mean: %s, accuracy: %s, RT_correct: %s, RT_wrong: %s' %( str(RT_mean_sub_bl), str(accuracy_sub_bl), str(RT_correct_bl), str(RT_wrong_bl))
# 	print 'bottom right: RT_mean: %s, accuracy: %s, RT_correct: %s, RT_wrong: %s' %( str(RT_mean_sub_br), str(accuracy_sub_br), str(RT_correct_br), str(RT_wrong_br))

# 	return np.array(all_responses), RT_mean_sub_br


# def cumulative_sum(x):
# 	return np.array([np.nansum(x[:i]) for i in range(len(x))]) #range(len(x))+1])

# for loop to get nansum of correct answers
	# accuracy_step = []
	# staircase = []
	# for i in len(all_responses):
	# 	cum_accuracy = np.nanmean(all_responses[:i])
		
	# 	accuracy_step.extend(item)
	# 	addition = np.nanmean(accuracy_step)
	# 	staircase.extend(addition)
	# pl.plot(staircase)

def plot_psychophysics():

	reaction_time, all_responses, task, button, position_x, position_y, trial_color, trial_ori, trial_stimulus = load_beh_data(csv_files)

	responses_mask = create_masks()[8] 

	red_task_mask = create_masks()[2]
	gre_task_mask = create_masks()[3]
	hor_task_mask = create_masks()[4]
	ver_task_mask = create_masks()[5]

	# shorter, becuase without nan. take the staircase values, regardless of correctness (compared with e.g.red_staircase), just throw away nan values
	red_staircase_all = np.abs(trial_color[responses_mask * red_task_mask])
	gre_staircase_all = np.abs(trial_color[responses_mask * gre_task_mask])
	hor_staircase_all = np.abs(trial_ori[responses_mask * hor_task_mask])
	ver_staircase_all = np.abs(trial_ori[responses_mask * ver_task_mask])

	#$shell()
	# calculate range
	red_min = red_staircase_all.min()
	red_max = red_staircase_all.max()
	
	bins = 6 #bins will be 6-1
	red_bin = np.linspace(red_min, red_max, bins, endpoint=True)

	# already got shorter, as cut the trials with nan values. use red staircase all
	red_bin1_mask =  (red_staircase_all> red_bin[0]) * (red_staircase_all <= red_bin[1]) 
	red_bin2_mask =  (red_staircase_all> red_bin[1]) * (red_staircase_all < (red_min+ red_bin[2])) 
	red_bin3_mask =  (red_staircase_all> red_bin[2]) * (red_staircase_all < (red_min+ red_bin[3])) 
	red_bin4_mask =  (red_staircase_all> red_bin[3]) * (red_staircase_all < (red_min+ red_bin[4])) 
	red_bin5_mask =  (red_staircase_all> red_bin[4]) * (red_staircase_all < (red_min+ red_bin[5])) 
	
	# do it like [] [] instead of [xxx * xxx], becuase the length should be the same
	ACC_red_bin1 = np.array(all_responses)[responses_mask * red_task_mask] [red_bin1_mask]
	ACC_red_bin2 = np.array(all_responses)[responses_mask * red_task_mask] [red_bin2_mask]
	ACC_red_bin3 = np.array(all_responses)[responses_mask * red_task_mask] [red_bin3_mask]
	ACC_red_bin4 = np.array(all_responses)[responses_mask * red_task_mask] [red_bin4_mask]
	ACC_red_bin5 = np.array(all_responses)[responses_mask * red_task_mask] [red_bin5_mask]

	f = pl.figure(figsize = (15,5))
	# color vs. ori on RT
	#s1 = f.add_subplot(141)
	
	objects = (red_bin[0], red_bin[1],red_bin[2],red_bin[3],red_bin[4])

	y_pos = np.arange(len(objects))
	y_values = np.array([np.mean(ACC_red_bin1), np.mean(ACC_red_bin2), np.mean(ACC_red_bin3), np.mean(ACC_red_bin4), np.mean(ACC_red_bin5)])
	sd = np.array([np.std(ACC_red_bin1), np.std(ACC_red_bin2), np.std(ACC_red_bin3), np.std(ACC_red_bin4), np.std(ACC_red_bin5)])
	n = np.array([np.array(ACC_red_bin1).shape[0], np.array(ACC_red_bin2).shape[0], np.array(ACC_red_bin3).shape[0], np.array(ACC_red_bin4).shape[0], np.array(ACC_red_bin5).shape[0] ])
	yerr = (sd/np.sqrt(n.squeeze()))*1.96
	
	#pl.f()
	#pl.errorbar(objects. y_values, yerr = yerr)
	pl.bar(y_pos, y_values, yerr = yerr, align = 'center', alpha = 0.5)
	pl.xticks (y_pos, objects, fontsize = 40) # why doesn't work?
	pl.title( 'color vs. ori on Accuracy')#, fontsize = 20)
	pl.ylim([0, 1])
	sn.despine(offset=10)

	pl.show()



def save_results (subname):
	# save results into a txt file
	#pl.savefig(figure_dir + 'scanner/staircase_plots/%s_color_ori_staircase_plot.pdf'%(subname))
	betas, t, p, t_Col_vs_Ori_RT, p_Col_vs_Ori_RT, t_Col_vs_Ori_Acc, p_Col_vs_Ori_Acc, t_cor_vs_incor_RT, p_cor_vs_incor_RT, t_righ_vs_wro_task_RT, p_righ_vs_wro_task_RT, RT_color, RT_ori, Accuracy_color, Accuracy_ori, RT_correct, RT_incorrect, RT_right_task, RT_wrong_task = compute_behavioral_performance(csv_files)

	sys.stdout = open(figure_dir + 'lab_%s_results.txt'%(subname), 'w')

		# f = open(os.path.join('data', self.subject_initials + '_staircase.txt'), 'w')
		# f.write(";".join([str(self.staircases[s].get_intensity()) for s in range(len(self.staircases))]))			
		# f.close()

	print '[main] Running analysis for %s'% (subname) #No. %s participant %s' % (str(subii+1), subname)
	print 'Results 1- GLM' 
	print '  taskvalue  distractor  interaction'
	print 'betas: %.2f %.2f %.2f' % (betas[0], betas[1], betas[2],)
	#print 'betas_new: %.2f %.2f %.2f'%  (betas_new[0], betas_new[1], betas_new[2])
	print 't-value: %.2f %.2f %.2f'% (t[0], t[1], t[2])
	print 'p-value: %.2f %.2f %.2f'%  (p[0], p[1], p[2])
	
	# Color vs Orientation on RT, %correct

	print 'Results 2 - Color vs Ori '
	print 'RT'
	print 't: %.2f ; p: %.2f' % (t_Col_vs_Ori_RT, p_Col_vs_Ori_RT)
	print 'Accuracy'
	print 't: %.2f ; p: %.2f' % (t_Col_vs_Ori_Acc, p_Col_vs_Ori_Acc)

	# Correct vs Incorrect task vs Incorrect direction on RT

	print 'Results 3 - Correct vs Incorrect answers on RT'
	print 't: %.2f ; p: %.2f' % (t_cor_vs_incor_RT, p_cor_vs_incor_RT)

	print 'Results 4 - Right vs Wrong tasks on RT'
	print 't: %.2f ; p: %.2f' % (t_righ_vs_wro_task_RT, p_righ_vs_wro_task_RT)


	# print 'F: %.2f ; p: %.2f ' % (F_answers_RT, p_answers_RT)
	# print 'posthoc tests, threashold: .05/3= .017'
	# print 'cor_vs_task', cor_vs_task
	# print 'cor_vs_dir', cor_vs_dir
	# print 'task_vs_dir',task_vs_dir

	sys.stdout.close()


	#GLM
	# s1 = f.add_subplot(231)

	# TASKVALUE[~np.isnan(reaction_time)]
	# pearsonr(TASKVALUE, RT)
	# pearsonr(DISTRACTOR, RT)
	f = pl.figure(figsize = (15,5))
	# color vs. ori on RT
	s1 = f.add_subplot(141)
	objects = ('color','orientation')
	y_pos = np.arange(len(objects))
	y_values = np.array([np.mean(RT_color), np.mean(RT_ori)])
	sd = np.array([np.std(RT_color), np.std(RT_ori)])
	n = np.array([np.array(RT_color).shape[0], np.array(RT_ori).shape[0] ])
	yerr = (sd/np.sqrt(n.squeeze()))*1.96
	pl.bar(y_pos, y_values, yerr = yerr, align = 'center', alpha = 0.5)
	pl.xticks (y_pos, objects, fontsize = 40) # why doesn't work?
	pl.title( 'color vs. ori on RT')#, fontsize = 20)
	pl.ylim([0, 1])
	sn.despine(offset=10)
	
	# color vs. ori on Accuracy
	s2 = f.add_subplot(142)
	objects = ('color','orientation')
	y_pos = np.arange(len(objects))
	y_values = np.array([np.mean(Accuracy_color), np.mean(Accuracy_ori)])
	sd = np.array([np.std(Accuracy_color), np.std(Accuracy_ori)])
	n = np.array([np.array(Accuracy_color).shape[0], np.array(Accuracy_ori).shape[0] ])
	yerr = (sd/np.sqrt(n.squeeze()))*1.96
	pl.bar(y_pos, y_values, yerr = yerr, align = 'center', alpha = 0.5)
	pl.xticks (y_pos, objects, fontsize = 40) # why doesn't work?
	pl.title( 'color vs. ori on Accuracy')#, fontsize = 20)
	pl.ylim([0, 1])
	sn.despine(offset=10)

	# correct vs. incorrect on RT
	s3 = f.add_subplot(143)
	objects = ('correct','incorrect')
	y_pos = np.arange(len(objects))
	y_values = np.array([np.mean(RT_correct), np.mean(RT_incorrect)])
	sd = np.array([np.std(RT_correct), np.std(RT_incorrect)])
	n = np.array([np.array(RT_correct).shape[0], np.array(RT_incorrect).shape[0] ])
	yerr = (sd/np.sqrt(n.squeeze()))*1.96
	pl.bar(y_pos, y_values, yerr = yerr, align = 'center', alpha = 0.5)
	pl.xticks (y_pos, objects, fontsize = 40) # why doesn't work?
	pl.title( 'correct vs. incorrect on RT')#, fontsize = 20)
	pl.ylim([0, 1])
	sn.despine(offset=10)

	# valid vs. invalid on RT
	s4 = f.add_subplot(144)
	objects = ('valid','invalid')
	y_pos = np.arange(len(objects))
	y_values = np.array([np.mean(RT_right_task), np.mean(RT_wrong_task)])
	sd = np.array([np.std(RT_right_task), np.std(RT_wrong_task)])
	n = np.array([np.array(RT_right_task).shape[0], np.array(RT_wrong_task).shape[0] ])
	yerr = (sd/np.sqrt(n.squeeze()))*1.96
	pl.bar(y_pos, y_values, yerr = yerr, align = 'center', alpha = 0.5)
	pl.xticks (y_pos, objects, fontsize = 40) # why doesn't work?
	pl.title( 'valid vs. invalid on RT')#, fontsize = 20)
	pl.ylim([0, 1])
	sn.despine(offset=10)

	pl.savefig(figure_dir + 'lab_%s_results.jpg'%(subname))


# loop over the subjects
for subii, subname in enumerate(sublist):

	#print '[main] Running analysis for No. %s participant %s' % (str(subii+1), subname)
	subject_dir= os.path.join(data_dir,subname)
	csv_files = glob.glob(subject_dir+'/*.csv')
	csv_files.sort()
	#shell()

	if csv_files[0].split('_')[2]=='0':
		csv_files.pop(0)

	#plot_staircase(csv_files,subname)
	#save_results(subname)
	plot_psychophysics()

	
# # not useful anymore
# 	runnum = str(subii+1)
# 	csv_filename = data_dir + subname + '/'+ subname.lower() +'_' + runnum + '_output.csv'
# 	csv_filename = "%s%s/%s_%d_output.csv" % (data_dir, subname, subname.lower() + runnum)











