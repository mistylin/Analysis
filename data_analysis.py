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
	''' compute RT, accuracy for each run, and averaged values over runs'''

	print ' RT  &  accuracy '
	reaction_time = []
	all_reponses = []
	# task = []
	# button =[]

	for this_file in csv_files: # loop over files

		csv_data = pd.read_csv(this_file)  #load data na_values = ["NaN"]
		
		RT_run = np.nanmean(csv_data['reaction_time'])
		accuracy_run = np.nanmean(csv_data['correct_answer'])

		reaction_time.extend(csv_data['reaction_time'])
		all_reponses.extend(csv_data['correct_answer'])
		# task.expend(csv_data['task'])
		# button.expend(csv_data['button'])
		
		print RT_run, accuracy_run

	RT_mean_sub = np.nanmean(reaction_time)
	accuracy_sub = np.nanmean(all_reponses)
	RT_correct = np.nanmean(reaction_time[all_reponses == 1])
	
	#wrong_direction = 
	#wrong_task = 
	#RT_wrong_direction = np.nanmean(reaction_time[all_reponses == 0])
	#RT_wrong_task = 
	print 'average', RT_mean_sub, accuracy_sub

	return np.array(all_reponses)


def cumulative_sum(x):

	return np.array([np.nansum(x[:i]) for i in range(len(x))])


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


	


	
# # not useful anymor
# 	runnum = str(subii+1)
# 	csv_filename = data_dir + subname + '/'+ subname.lower() +'_' + runnum + '_output.csv'
# 	csv_filename = "%s%s/%s_%d_output.csv" % (data_dir, subname, subname.lower() + runnum)




"""calculate 1)reaction_time 2)correct_rate for different condiitons
(correct response, wrong feature, wrong direction)"""


"""plot the staircase--correct rate over time"""









