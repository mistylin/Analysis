import numpy as np
import math
import matplotlib.pyplot as plt

def plot_point(point, angle, length):

	'''
	point - Tuple (x, y)
	angle - Angle you want your end point at in degrees.
	length - Length of the line you want to plot.

	Will plot the line on a 10 x 10 plot.
	'''

	# unpack the first point
	x, y = point

	# find the end point
	endy = length * math.sin(math.radians(angle))
	endx = length * math.cos(math.radians(angle))

	startx = -endx
	starty = -endy

	# plot the points
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.set_ylim([-3, 3])   # set the bounds to be 10, 10
	ax.set_xlim([-3, 3])
	ax.plot([startx, endx], [starty, endy])

	fig.savefig( 'test_lines.png')

	#fig.show()



plot_point((0,0), 22.5, 2)


def plot_point(point, angle, length):
     '''
     point - Tuple (x, y)
     angle - Angle you want your end point at in degrees.
     length - Length of the line you want to plot.

     Will plot the line on a 10 x 10 plot.
     '''

     # unpack the first point
     x, y = point

     # find the end point
     endy = length * math.sin(math.radians(angle))
     endx = length * math.cos(math.radians(angle))

     # plot the points
     fig = plt.figure()
     ax = plt.subplot(111)
     ax.set_ylim([0, 10])   # set the bounds to be 10, 10
     ax.set_xlim([0, 10])
     ax.plot([x, endx], [y, endy])

     fig.show()



def run_regression(fileii, design_matrix, design_matrix_selection, fmri_data, regression = 'RidgeCV'):

	global n_voxels, n_TRs, n_regressors, df #, results, r_squareds, alphas, intercept, betas, _sse 

	n_voxels = fmri_data.shape[0]
	n_TRs = fmri_data.shape[1]
	n_regressors = design_matrix.shape[1]
	df = (n_TRs-n_regressors)

	results = np.zeros((n_voxels,3))
	r_squareds =  np.zeros((n_voxels, )) 
	alphas =  np.zeros((n_voxels, 1))
	intercept =  np.zeros((n_voxels, 1))
	betas = np.zeros((n_voxels, n_regressors ))  #shape (5734, 71)
	_sse = np.zeros((n_voxels, ))

	r_squareds_selection =  np.zeros((n_voxels, ))
	betas_selection = np.zeros((n_voxels, n_regressors ))

	print 'n_voxels without nans', n_voxels
# # GLM to get betas

	if regression == 'GLM': #'RidgeCV'
		print 'start %s GLM fitting'%(str(fileii))
		betas, _sse, _r, _svs = np.linalg.lstsq(design_matrix, fmri_data.T )
		# betas shape (65, 9728--number of voxels)
		# _sse shape (10508)
		r_squareds = 1.0 - ((design_matrix.dot(betas).T -fmri_data)**2).sum(axis=1) / (fmri_data**2).sum(axis=1)
		betas = betas.T #(10508,72)

		# for selection
		# betas_selection, _sse_selection, _r_selection, _svs_selection = np.linalg.lstsq(design_matrix_selection, fmri_data.T )
		# r_squareds_selection = 1.0 - ((design_matrix_selection.dot(betas_selection).T -fmri_data)**2).sum(axis=1) / (fmri_data**2).sum(axis=1)
		print 'finish GLM'

	elif regression == 'RidgeCV':
		# ridge_fit = RidgeCV(alphas = np.linspace(1,500,500) , fit_intercept = False, normalize = True )
		ridge_fit = RidgeCV(alphas = np.linspace(1,400,400) , fit_intercept = False, normalize = True )		
		# alpha_range = [0.001, 1000]
		#[0.001,0.01,1,10,100,1000]
		#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
		# alpha_range = [0.5]
		# alpha_range = np.concatenate((np.array([0.001,0.01]), np.linspace(0.1,0.5,4, endpoint =False), np.linspace(0.5,10,19, endpoint =False), np.linspace(10,1000,100, endpoint =True)  ) )
		# alpha_range = np.concatenate((np.array([0.001,0.01,0.1]), np.linspace(0.5,10,19, endpoint =False), np.linspace(10,1000,100, endpoint =True)  ) )
		# alpha_range = np.concatenate((np.array([0.01,0.1]), np.linspace(0.5,10,19, endpoint =False), np.linspace(10,1000,100, endpoint =True)  ) )
		# alpha_range = np.concatenate((np.array([0.1]), np.linspace(0.5,10,19, endpoint =False), np.linspace(10,1000,100, endpoint =True)  ) )
		# alpha_range = np.concatenate( (np.linspace(0.5,10,19, endpoint =False), np.linspace(10,1000,100, endpoint =True)  ) )
		# alpha_range = np.concatenate( (np.linspace(1,10,18, endpoint =False), np.linspace(10,1000,100, endpoint =True)  ) )
		# alpha_range = np.concatenate( (np.array([1.0]), np.linspace(10,1000,100, endpoint =True)  ) )

		# ridge_fit = RidgeCV(alphas = alpha_range , fit_intercept = True, normalize = True)
		
		# ridge_fit_selection = RidgeCV(alphas = alpha_range , fit_intercept = True, normalize = True)

		
		print 'start %s RidgeCV fitting'%(str(fileii))
		
		for x in range(n_voxels):
			
			ridge_fit.fit(design_matrix, fmri_data[x, :])
			print x, ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_ #, ridge_fit.coef_.T
			# results[x] = [ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_, ridge_fit.coef_.T]	#ridge_fit.fit(design_matrix, fmri_data.T)
			r_squareds[x] = ridge_fit.score(design_matrix, fmri_data[x,:])
			alphas[x] = ridge_fit.alpha_
			betas[x] = ridge_fit.coef_.T
			intercept[x,:] = ridge_fit.intercept_
			_sse[x] = np.sqrt(np.sum((design_matrix.dot(betas[x]) - fmri_data[x,:])**2)/df)

			# # for selection
			# ridge_fit_selection.fit(design_matrix_selection, fmri_data[x,:])
			# r_squareds_selection[x] = ridge_fit_selection.score(design_matrix_selection, fmri_data[x,:])
			# betas_selection[x] = ridge_fit.coef_.T

		print 'finish RidgeCV'

	return r_squareds, r_squareds_selection, betas, betas_selection, _sse, intercept, alphas













