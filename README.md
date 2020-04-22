Jigsaw Multilingual Toxic Comment Classification
==============================

This github repo contains Jigsaw Multilingual Toxic Comment Classification. This is a public competition task on `kaggle.com`. Hence was my choice for the Udacity Machine Learning Engineer Nanodegree Capstone Project. It contains all the code used in the project.

Setup Instructions
-------------
To download, please use the following command:
```
git clone https://github.com/nwachukwu-anthony/Toxic-Comment-Classification.git
```

This will download the repo to the current directory. To ensure that you have the necessary packages, please use the Anaconda distribution of Python. With Anaconda, you will have access to the conda environment manager. In the newly created folder, type:
```
conda env create -f environment.yml
```

This will create a new environment named _capstone_env_. Please note that it will download a bunch of packages. To activate the environment:
*  Windows: ```activate capstone_env```
*  macOs and Linux: ```source activate capstone_env```

After this has been completed, you can activate Jupyter Notebook by typing ```jupyter notebook``` into your shell. You can then explore the notebooks. The notebooks are ordered and should be run in sequence, more or less. If you run a later notebook without having run an earlier one, it is possible that an exception will be raised since some notebooks rely on the outputs of previous notebooks.

You will need to install these packages to be able to use the Notebook. You can install them via Notebook or terminal. In the later case, you need to omit **!**.

`!pip install googletrans  
!pip install tqdm  
!pip install torch  
!pip install emoji  
!pip install --upgrade pip  
!pip install translators`  


Project Organization
------------

    |--- LICENSE
    |--- README.md		<- The top-level README for developers using this project.
    |--- data			<- When the notebook is being executed, this folder is created to store the data being generated.
    |--- notebooks		<- The control of the project. All information and calls to other folders and external functions are made here.
    |--- serve
	|--- model.py		<- The neural network for training the data is contained here. To be used by for deployment.
	|--- predict.py		<- Makes predictions and sends result to the website.
	|--- requiremants.txt	<- Packages used for the prediction is found here to be used for deployment.
	|--- utils.py		<- Functions to be called by `predict` to be used for data preparation.
    |--- train
	|--- model.py		<- The neural network for training the data is contained here.
	|--- requirements.txt	<- Packages used for the model training is found.
	|--- train.py		<- Trains the model by making calls to it.
    |--- website
	|--- index.html		<- Accepts input from user, sends to the `predict` which returns result to it for the user.
    |--- helper


--------

<p><small>Link to data <a target="_blank" href="https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/">kaggle Jigsaw Data</a>.</small></p>
