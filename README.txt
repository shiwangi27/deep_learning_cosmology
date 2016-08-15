DATASETS : 

There are three different datasets from different Theoretical models used for the Experiments. For the most part, we used fico_z02 dataset. To compare this with other Theoretical models, we used si75_z02 and si85_z02 datasets. 

Flattened 1000 Images of 1024 x 1024 

fico -- /global/homes/s/ssingh79/data/conv_z02.h5     OR 
        /global/cscratch1/sd/ssingh79/fico_data/conv_z02.h5        
si75 -- /global/cscratch1/sd/ssingh79/si75_data/si75_z02.h5
si85 -- /global/cscratch1/sd/ssingh79/si85_data/si85_z02.h5 

Flattened 64000 Images of 128 x 128

fico -- /global/homes/s/ssingh79/data/train_data_64k.h5     OR       
        /global/cscratch1/sd/ssingh79/fico_data/train_data_64k.h5
        
si75 -- /global/homes/s/ssingh79/data/si75_train_data_64k.h5   OR 
        /global/cscratch1/sd/ssingh79/si75_data/si75_train_data_64k.h5
        
si85 -- /global/cscratch1/sd/ssingh79/si85_data/si85_train_data_64k.h5

--------------------------------------------------------------------------------------------------------------------------

CODE DIRECTORY STRUCTURE : 

The most important directory is /global/homes/s/ssingh79/convolutional_autoencoder-master/Models

Following are the models that has the good results:
-----------------------------------------------------

1. To see the Features learned with a denoised CAE on two different datasets:
    Denoising Convolutional Autoencoder -- Validate_DCAE/Denoising_CAE_validation.py AND
                                           Validate_DCAE/Denoising_CAE_valid_si75.py
            
2. To see if the model does a good job on two Theoretical models in an unsupervised learning: fico_z02 & si85_z02
    Unsupervised DCAE -- Unsupervised_theoretical/Unsupervised_DCAE.py
            
3. To see if the model does well on two Theoretical models in a supervised learning: fico_z02 & si75_z02
    Supervised CNN -- Theoretical_model/supervised_cnn.py 
                      Theoretical_model/supervised_cnn_2.py  (Only difference is this one has two dense layers) 

------------------------------------------------------

All the discarded models are stored in the ../Model/Conv_ae_models/ dircetory right from the start! 

In addition to the above files, there are other supporting python codes: 
1. shape.py 
2. visualize.py
3. load_theoretical_models.py
4. guided_backprop.py
5. embed_tsne.py

---------------------------------------------------------------------------------------------------------------------------

OUTPUT FILES LOCATION : 

The parent directory for all the Output files is /global/homes/s/ssingh79/convolutional_autoencoder-master/output_files
Most of the output file names are after the sample size (ss) and learing rate (lr) preceded by the name of the model. 

Following are the output folder names if you need to see the results:
----------------------------------------------------------------------

1. Denoising Convolutional Autoencoder -- Valid_lr_0.05_ss_10000
                                          Valid_si75_lr_0.05_ss_10000
                                       
2. Unsupervised DCAE -- Unsupervised_ss_1000_lr_5e-2

3. Supervised CNN -- Supervised_ss_1000_lr_3e-4

-----------------------------------------------------------------------

Files in the output folder: 
1. Learning curve, validation curve
2. Saved weights - model.npz 
3. Denselayer weights - bottleneck_layer.npz
4. Saved loss, val_loss - reconst_error.npz, reconst_error_valid.npz 
5. Input images - normalized_maps.h5
6. Reconstructed images - reconstructed_maps.h5
7. Saliency maps - saliency_maps.h5
8. Tsne plots and embedded tsne plots
9. Others are some normalized & reconstructed images. 

You can pretty much ignore other output files in the dir. Most of the output files will have a archive where discarded results are saved. Same way with the Slurm files, there is slurm archive file. Ignore! Focus on the above directories Names! 

-------------------------------------------------------------------------------------------------------------------------

GitHub link : https://github.com/shiwangi27/deep_learning_cosmology 
 





