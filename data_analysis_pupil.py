# Pupil analysis

import os,sys

from IPython import embed 

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

sys.path.append( 'utils' )

# Import hedfpy functions
# from hedfpy import EDFOperator
# from hedfpy import EyeSignalOperator
# from hedfpy import HDFEyeOperator

# Martijns analysis tools
# import Analyzer
# from BehaviorAnalyzer import BehaviorAnalyzer
from PupilAnalyzer import PupilAnalyzer


subID = 'ms'

data_folder = '/home/barendregt/Analysis/xiaomeng/Data/'


h5filename = os.path.join(data_folder + '/' + subID + '.h5')

pa = PupilAnalyzer(subID, h5filename, data_folder)

pa.load_data()

embed()

pa.get_aliases()

alias = pa.aliases[0]

trial_time_info = pa.h5_operator.read_session_data(alias, 'trials')

run_time_info  = pa.h5_operator.read_session_data(alias,'blocks')

run_start_time = run_time_info['block_start_timestamp'][0]
run_end_time = run_time_info['block_end_timestamp'][0]

pupil_signal = pa.h5_operator.signal_during_period(time_period = [run_start_time, run_end_time], alias = alias, signal = 'pupil_bp_clean_', requested_eye = pa.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0])
