# About
This is a project written by Jeremy Tsai (UCLA CS '23) during the Cornell NeuroNex 2021 Program under the instruction/mentorship of Professor Mert Subuncu and Zijin Gu. The project's goal is to build a group-level encoding model that predicts how a particular image will activate the brain on average for a given Region of Interest (ROI). The model would be trained using the NSD data.  
# Setting Up
Include conda virtual environment information here.
The following is essentially identical to the `conda` documentation (https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
1. Create the virtual environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
    The first line of the `environment.yml` file sets the new environment's name. It is currently set to `avg-model`.  

2. Activate the new environment through: `conda activate myenv`. If `environment.yml` has not been tampered with, `myenv` should be `avg-model`.  

3. Verify that the new enviroment was installed correctly:
    ```bash
    conda env list
    ```
    You should see something like:
    ```bash
    myenv             *  /opt/anaconda3/envs/avg-model
    ```
    Again, if `environment.yml` has not been tampered with, `myenv` should be `avg-model`.
# Acknowledgements
This project is heavily derived from the work of my mentor, Zijin Gu. Refer to her original work (NeuroGen) here: https://github.com/zijin-gu/NeuroGen. Here is a link to the paper documenting her work: https://arxiv.org/abs/2105.07140#:~:text=NeuroGen%20combines%20an%20fMRI%2Dtrained,of%20macro%2Dscale%20brain%20activation.
# Training Data
Part of the training data are listed in Zijin's aforementioned GitHub NeuroGen repository, under the file `roiavgbeta`. These are the activation (a single float number) of the particular ROI of the particular subject. The image that causes this activation is too big to be uploaded to GitHub. Contact Jeremy for this data. Note that even with these raw `.h5py` files are not sorted in the order they are presented to the subjects. These sorting information are stored in the `nsd_expdesign.mat` file, which is also omitted from this GitHub repository because of concerns that NSD data has not been made publicly available yet.  
There is a relatively small number of images that are shown to all 8 subjects of the NSD data set. These shared images can be easily used to train the data, because there is a clear correct label (i.e. the average brain activation of all 8 subjects to the specific shared image). There are also images that are only shown to a few subjects, but not all 8 of them. We call these images the partially-shared images. The way we are using to utilize these data is: **STILL UNDECIDED**. Similarly, images that are only shown to one individual (referred to as the _unique pictures_) are also utilized in an **UNDECIDED** way.  
# Training Process
# Roadmap/Todos
- Implement a command line interface to specify the network architecture (e.g. Alex-Net-like CNN, fully connected ANN, etc.) and optimization techniques (e.g. Adam, SGD, and hyperparameters). Status: **SOMEWHAT COMPLETED**.
- Implement a logging system for effective debugging. Status: **SETUP COMPLETED**.
- Figure out the data distribution and what to do with them (w.r.t. to shared images, partially shared images, and unique images). Status: **COMPLETED, decided to do imputation.**
- Brief preliminary visualization of training data. Status: **MOSTLY COMPLETED**. This task was carried out to determine if those pictures that the subject has seen multiple time are spread out significantly. If they are not spread out too much (i.e. the activations are consistent), we can just take the mean, which can reduce the training samples by 3 times. The relevant graphs generated are in the `graphs` folder. The results showed that the data are quite spread out, and the spread of activation is similar in all subjects whether it is the shared image set or the entire image set. Furthermore, as per Zijin's advice, it is desirable to get all response as the training data to prevent underfitting.
- Imputation of data through kNN + SVD. Status: **INCOMPLETE**.