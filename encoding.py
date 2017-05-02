# just a model, a structure to be used to fit the raw data to get best parameters.

# use meshigrid function
# dot (cube * timecourse)



''' first, extracted and averaged the (normalized) responses of each
voxel in each visual area or searchlight neighborhood (see below, Searchlight
analysis) over a period from 6–10 s after the start of each trial.'''


'''next, devide data into training & test sets. 
for the training set data, we modeled the measured responses of each voxel in the training set
as a linear sum of nine orientation “channels,” each with an idealized
response function.'''


# normalize BOLD



# sort trials into 64 bins (8*8)



# devide into 2 dataset: training, 


# model- basic function/ channel responses


# linear sum of 64 basic models

design_matrix = np. array()

design_matrix = np.array([L_model_data_at_TRs, R_model_data_at_TRs, np.ones(R_model_data_at_TRs.shape[0])])

fmri_data = np.array([left_voxel_timecourse_Z,right_voxel_timecourse_Z]).squeeze()
betas, _sse, _r, _svs = np.linalg.lstsq(design_matrix.T, fmri_data.T )

## however, the models here are still the timecourse







