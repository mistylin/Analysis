import pandas as pd
from math import *
import numpy as np
import os
import glob

data_dir = '/Users/xiaomeng/disks/Aeneas_Raw/2017/visual/Attention/Behavioural/'
sublist =  ['SL','MS']

def compute_behavioral_performance:
	reaction_time = data['reaction_time']
	response = data['correct_answer']
	RT_mean_run = np.mean(reaction_time)
	accuracy_run = np.mean(response)

	RT_mean_sub = 
	accuracy_sub =

	




# loop over the subjects
for subname in sublist:
	subject_dir= os.path.join(data_dir,subname)
	csv_files = glob.glob(subject_dir+'*.csv')

	for this_file in csv_files:
		data = pd.read_csv(this_file, na_values = ["."])  #load data

	


	
# not useful anymor
	runnum = str(subii+1)
	csv_filename = data_dir + subname + '/'+ subname.lower() +'_' + runnum + '_output.csv'
	csv_filename = "%s%s/%s_%d_output.csv" % (data_dir, subname, subname.lower() + runnum)




#def load_data():
		
data = pd.read_csv(csv_filename, na_values = ["."])



# calculate average reaction time


print RT_mean

# loop over the runs, subjects. name them differently

"""calculate 1)reaction time per run, 2)averaged reaction time per person"""

"""compute performance attended"""

"""calculate 1)reactiont_time 2)correct_rate for different condiitons
(correct response, wrong feature, wrong direction)"""


"""plot the staircase--correct rate over time"""









