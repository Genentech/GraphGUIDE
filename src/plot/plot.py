import numpy as np
import matplotlib.pyplot as plt

def plot_mnist_digits(
	digits, grid_size=(10, 1), scale=1, clip=True, title=None
):
	"""
	Plots MNIST digits. No normalization will be done.
	Arguments:
		`digits`: a B x 1 x 28 x 28 NumPy array of numbers between
			0 and 1
		`grid_size`: a pair of integers denoting the number of digits
			to plot horizontally and vertically (in that order); if
			more digits are provided than spaces in the grid, leftover
			digits will not be plotted; if fewer digits are provided
			than spaces in the grid, there will be at most one
			unfinished row
		`scale`: amount to scale figure size by
		`clip`: if True, clip values to between 0 and 1
		`title`: if given, title for the plot
	"""
	digits = np.transpose(digits, (0, 2, 3, 1))
	if clip:
		digits = np.clip(digits, 0, 1)
	
	width, height = grid_size
	num_digits = len(digits)
	height = min(height, num_digits // width)
	
	figsize = (width * scale, (height * scale) + 0.5)
	
	fig, ax = plt.subplots(
		ncols=width, nrows=height,
		figsize=figsize, gridspec_kw={"wspace": 0, "hspace": 0}
	)
	if height == 1:
		ax = [ax]
	if width == 1:
		ax = [[a] for a in ax]
	
	for j in range(height):
		for i in range(width):
			index = i + (width * j)
			ax[j][i].imshow(digits[index], cmap="gray", aspect="auto")
			ax[j][i].axis("off")
	if title is not None:
		ax[0][0].set_title(title)
	plt.subplots_adjust(bottom=0.25)
	plt.show()
