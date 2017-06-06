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