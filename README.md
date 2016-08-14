# deep_learning_cosmology
<b> Deep Learning on Cosmology Maps </b>
This repo includes following model coded in Theano and Lasagne working at NERSC@Berkeley Labs. We ran code on <a href="http://www.nersc.gov/users/computational-systems/cori/">Cori</a>. </br>
Denoising Convolutional Autoencoders </br>
Supervised Convolutional NN </br>

Also includes -- </br> 
Visualization of features learned in Deep Learning </br> 

To run on Cori, navigate to the model dir and do : </br>
module load deeplearning </br>
python <model_name>.py </br>

To run on Edison, navigate to the model dir and do : </br>
module load python </br>
module load scikit-learn </br> 
module load h5py </br>
module load theano </br>
python <model_name>.py </br>

To run the code in General : </br> 
Have the deeplearing environment with the following installed : </br>
1. python3 </br>
2. theano </br>
3. lasgane </br>
4. scikit-learn </br>
5. h5py </br>
6. numpy </br>
7. matplotlib </br>

References : 

