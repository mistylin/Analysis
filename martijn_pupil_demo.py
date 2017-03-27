# Pupil analysis

import os,sys

from IPython import embed as shell

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

sys.path.append( 'utils' )

# Martijns analysis tools
from PupilAnalyzer import PupilAnalyzer # from the file import the method, the file is in the folder named 'utils'

# data_folder = '/home/barendregt/Analysis/xiaomeng/Data/'
# figure_dir = '/home/barendregt/Analysis/xiaomeng/'

data_folder = '/home/xiaomeng/Data/Pre_scan_data/'
figure_dir = '/home/shared/2017/visual/Attention/eye/'

deconvolution_interval = np.array([-1.5, 4.5])
down_fs = 100


sublist = ['xy'] #'MS', 'SL']

for subID in sublist:
	
	print '[main] Running analysis for %s' % (subID)

	sub_data_folder = os.path.join(data_folder, str(subID) + '/') #data_folder

	h5filename = os.path.join(sub_data_folder, subID + '.h5')

	##
	## The reference_phase argument determines the anchor for the epoching
	## 0 = ITI
	## 1 = Task?
	## 2 = interval
	## 3 = Stimulus 1
	## 4 = ISI
	## 5 = Stimulus 2
	## 6 = Response
	## 

	## 6 = Response phase - where the question mark 
	## 7 = Button press
	### while right now 6 = button press
	pa = PupilAnalyzer(subID, h5filename, sub_data_folder, signal_downsample_factor = down_fs, deconvolution_interval = deconvolution_interval) # create an object

	##
	## Example function calls:
	## reference_phase = which trial phase to time-lock to
	## with_rt = add reaction time or not (default is not)
	##
	## pa.signal_per_trial(reference_phase = 3) is equivalent to:
	## pa.signal_per_trial(reference_phase = 3, with_rt = False)
	##
	pa.signal_per_trial(reference_phase = 3, with_rt = False)

	trial_signal_sub_mean = np.mean(pa.trial_signals, axis =0) 

	#shell()
	f = plt.figure(figsize = (8,6))
	plt.plot(trial_signal_sub_mean)
	plt.savefig(figure_dir + '%s_mean_pupil_size_plot.jpg'%(subID))

	print 'finish one participant !!!'
	#plt.show()


