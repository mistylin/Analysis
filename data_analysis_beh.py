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

sn.set_style('ticks')

run_on_aeneas = False

if run_on_aeneas:
	data_dir = '/home/raw_data/2017/visual/Attention/fMRI/'
	figure_dir = '/home/shared/2017/visual/Attention/behaviour/'
else:
	data_dir = '/Users/xiaomeng/disks/Aeneas_Raw/2017/visual/Attention/fMRI/'
	figure_dir = '/Users/xiaomeng/disks/Aeneas_Shared/2017/visual/Attention/behaviour/'

sublist =  ['DE','TK','MV']
beh ='beh/'

'''fMRI beh >> 1)change sublist, 2)csv_files path, 3)buttons 4) savefig folders 5) locaitons 6) pop out csv_files with 0 7)'''

def load_data(csv_files):
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
	reaction_time, all_responses, task, button, position_x, position_y, trial_color, trial_ori, trial_stimulus = load_data(csv_files) #  all the data

	
	color_task_mask = np.array(np.array(task)==1)
	ori_task_mask = np.array(np.array(task)==2)

	red_task_mask = np.array(color_task_mask * ((trial_stimulus == 0) + (trial_stimulus == 1)))
	gre_task_mask = np.array(color_task_mask * ((trial_stimulus == 2) + (trial_stimulus == 3)))
	hor_task_mask = np.array(ori_task_mask * ((trial_stimulus == 0) + (trial_stimulus == 2)))
	ver_task_mask = np.array(ori_task_mask * ((trial_stimulus == 1) + (trial_stimulus == 3)))
	
	right_task_mask = np.array(((np.array(task)==1)*((np.array(button)=='w')+(np.array(button)=='e')))+((np.array(task)==2)*((np.array(button)=='b')+(np.array(button)=='y')))) # sfjl
	wrong_task_mask = np.array(((np.array(task)==2)*((np.array(button)=='w')+(np.array(button)=='e')))+((np.array(task)==1)*((np.array(button)=='b')+(np.array(button)=='y')))) # sfjl

	responses_mask = np.array((np.array(all_responses)==1) + ((np.array(all_responses)==0)* right_task_mask)) 

	correct_answer_mask = np.array(np.array(all_responses)==1) 
	incorrect_answer_mask = np.array((np.array(all_responses)==0) * (~np.array(wrong_task_mask)))

	 #four locations
	top_left_mask = np.array((np.array(position_x) == -1.5) * (np.array(position_y) == 1.5)) # 2.5
	top_right_mask = np.array((np.array(position_x) == 1.5) * (np.array(position_y) == 1.5))
	bottom_left_mask = np.array((np.array(position_x) == -1.5) * (np.array(position_y) == -1.5))
	bottom_right_mask = np.array((np.array(position_x) == 1.5) * (np.array(position_y) == -1.5))

	return color_task_mask, ori_task_mask, red_task_mask, gre_task_mask, hor_task_mask, ver_task_mask, right_task_mask, wrong_task_mask, responses_mask, correct_answer_mask, incorrect_answer_mask, top_left_mask, top_right_mask, bottom_left_mask, bottom_right_mask 


def plot_staircase(csv_files, subname):
	'''plot stairecase for each participant'''
	
	all_responses = load_data(csv_files)[1] # index starts from 0
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
	trial_color = load_data(csv_files)[6]
	trial_ori = load_data(csv_files)[7]

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
	y_values = np.array([np.mean(red_task_accuracy[51:]), np.mean(gre_task_accuracy[51:]), np.mean(hor_task_accuracy[51:]), np.mean(ver_task_accuracy[51:])])
	#shell()
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
	pl.savefig(figure_dir + 'scanner_%s_color_ori_staircase_plot.pdf'%(subname))
	#pl.savefig('/Users/xiaomeng/disks/Aeneas_Home/pdfs/%s_color_ori_staircase_plot.pdf'%(subname))

def compute_behavioral_performance(csv_files):
	''' compute 
	1) RT, accuracy for each run, and averaged values over runs
	2) RT for different condiitons 
	(correct response, ,wrong response, wrong task, wrong direction)'''


	reaction_time, all_responses, task, button, position_x, position_y, trial_color, trial_ori, trial_stimulus = load_data(csv_files)

	color_task_mask, ori_task_mask, red_task_mask, gre_task_mask, hor_task_mask, ver_task_mask, right_task_mask, wrong_task_mask, responses_mask, correct_answer_mask, incorrect_answer_mask, top_left_mask, top_right_mask, bottom_left_mask, bottom_right_mask = create_masks()

	RT_color = np.array(reaction_time)[color_task_mask] # with nan values
	RT_ori = np.array(reaction_time)[ori_task_mask] # with nan values
	Accuracy_color = np.array(all_responses)[color_task_mask] # with nan values
	Accuracy_ori = np.array(all_responses)[ori_task_mask] # with nan values

	RT_correct = np.array(reaction_time)[correct_answer_mask]
	RT_incorrect = np.array(reaction_time)[incorrect_answer_mask]
	RT_right_task = np.array(reaction_time)[right_task_mask]
	RT_wrong_task = np.array(reaction_time)[wrong_task_mask]

	# GLM to test RT vs TASKVALUE or DISTRACTOR or interaction
	norm_color = np.abs(np.array(trial_color)) / np.abs(np.array(trial_color)).max()
	norm_ori = np.abs(np.array(trial_ori)) / np.abs(np.array(trial_ori)).max()

	TASKVALUE = np.zeros((len(reaction_time),))
	TASKVALUE[np.array(task)==1] = norm_color[np.array(task)==1]
	TASKVALUE[np.array(task)==2] = norm_ori[np.array(task)==2]
	TASKVALUE = TASKVALUE[~np.isnan(reaction_time)] #* ~np.isnan(all_responses)] # why use two creteria?

	DISTRACTOR = np.zeros((len(reaction_time),))
	DISTRACTOR[np.array(task)==2] = norm_color[np.array(task)==2] #how do you define the distractor?
	DISTRACTOR[np.array(task)==1] = norm_ori[np.array(task)==1]
	DISTRACTOR = DISTRACTOR[~np.isnan(reaction_time)]# * ~np.isnan(all_responses)]

	RT = np.array(reaction_time)[~np.isnan(reaction_time)]# * ~np.isnan(all_responses)]
	X = np.hstack([np.ones((len(RT),1)), TASKVALUE[:,np.newaxis], DISTRACTOR[:,np.newaxis], TASKVALUE[:,np.newaxis]*DISTRACTOR[:,np.newaxis]])
	betas = np.linalg.lstsq(X, RT)[0]
	#betas_new = np.linalg.pinv(X).dot(RT)
	SE = np.sqrt(np.sum((X.dot(betas) - RT)**2)/(RT.size - X.shape[1]))
	df= RT.size - X.shape[1]
	t = [betas.squeeze().dot(contrast) / SE for contrast in np.array([[0,1,0,0],[0,0,1,0], [0,0,0,1]])]
	p = scipy.stats.t.sf(np.abs(t), df)*2


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

	return betas, t, p, t_Col_vs_Ori_RT, p_Col_vs_Ori_RT, t_Col_vs_Ori_Acc, p_Col_vs_Ori_Acc, t_cor_vs_incor_RT, p_cor_vs_incor_RT, t_righ_vs_wro_task_RT, p_righ_vs_wro_task_RT 

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


def save_results (subname):
	# save results into a txt file
	#pl.savefig(figure_dir + 'scanner/staircase_plots/%s_color_ori_staircase_plot.pdf'%(subname))
	betas, t, p, t_Col_vs_Ori_RT, p_Col_vs_Ori_RT, t_Col_vs_Ori_Acc, p_Col_vs_Ori_Acc, t_cor_vs_incor_RT, p_cor_vs_incor_RT, t_righ_vs_wro_task_RT, p_righ_vs_wro_task_RT = compute_behavioral_performance(csv_files)

	sys.stdout = open(figure_dir + 'scanner_%s_results.txt'%(subname), 'w')

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


# loop over the subjects
for subii, subname in enumerate(sublist):

	#print '[main] Running analysis for %s'% (subname) #No. %s participant %s' % (str(subii+1), subname)

	subject_dir= os.path.join(data_dir,subname,beh)

	csv_files = glob.glob(subject_dir+ '/*.csv')

	csv_files.sort()
	#shell()

	if csv_files[0].split('_')[2]=='0':
		csv_files.pop(0)

	plot_staircase(csv_files,subname)
	save_results(subname)




	
# # not useful anymore
# 	runnum = str(subii+1)
# 	csv_filename = data_dir + subname + '/'+ subname.lower() +'_' + runnum + '_output.csv'
# 	csv_filename = "%s%s/%s_%d_output.csv" % (data_dir, subname, subname.lower() + runnum)











