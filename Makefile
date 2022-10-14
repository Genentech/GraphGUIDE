install-dependencies:
	# Must be running Python 3.8
	conda install -y -c pytorch pytorch=1.11.0
	TORCH=$(python -c "import torch; print(torch.__version__)")
	CUDA=$(python -c "import torch; print(torch.version.cuda)")
	
	pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
	pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html  # This installs NumPy and SciPy
	
	pip install torch-geometric  # This installs tqdm and scikit-learn
	
	pip install networkx==2.6
	conda install -y -c anaconda click pymongo jupyter pandas
	conda install -y -c conda-forge matplotlib
	pip install sacred tables vdom
	conda install -y h5py

# Note about torch-scatter:
# It is possible that this installation pipeline will not install torch-scatter
# successfully. In particular, torch-scatter might import fine, but it may not
# be usable on GPU. Using certain PyTorch Geometric operations might cause a
# "Not compiled with CUDA support" error to be thrown on GPU. To fix this,
# reinstall torch-scatter as follows:
# 1. Navigate to https://data.pyg.org/whl/, which has all the pip wheel files
# 	for PyTorch Geometric
# 2. Go to https://data.pyg.org/whl/torch-1.11.0+cu113.html
# 	Replace the versions of PyTorch and CUDA appropriately, using the same
# 	versions as $TORCH and $CUDA above; for example, cu113 is for CUDA 11.3
# 3. Download the wheel file `torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl`,
# 	which is for Linux; the `cp38` refers to Python 3.8, which is what we're
# 	using
# 4. Uninstall torch-scatter using `pip uninstall torch-scatter` and manually
#	install the wheel file using `pip install torch_scatter*.whl`
