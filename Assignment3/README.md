# Assignment 3 CS5500

It includes the solution to all questions of the Assignment 3 of CS55500 Deep Learning Course

# Installation Instructions

1. Unzip the file **cs23mtech15001.zip** in a directory
2. Please create a virtual environment using the command `virtualenv venv` and use `source bin/activate` to activate the same
3. Install the required libraries using the command `pip install -r requirements.txt`
4. Install the required gym environments 
	1. pip install "gym[atari]"
	2. pip install "gym[accept-rom-license]"
	to install the atari game as it does not gets installed by default.

5. Launch the jupyter notebook by executing `jupyter notebook` in the current directory


Note:

# Question 1
Separate folders are present for each environment 

	a. MountainCarV0
		1. mountaincar_dql_45243.pt  - Best trained model
		2. MountainCarV0.ipynb - Jupyter notebook having all the code implementation for Q1
	b. PongV0
		1. 900th_epoch_model - folder containing the best trained model
		2. out.txt - file containing command line outputs of training
		3. pong.py - Python script to train for PongV0 environment
		4. Q1_PongV0.ipynb - Corresponding Jupyter notebook for answering Q1

# Question 2
	a. Q2.py -  Python script which can run from command line
	b. Q2.ipynb - Jupyter notebook having compilation of all experiments



# Owner
Abhinav Kumar Jha (cs23mtech15001@iith.ac.in)
