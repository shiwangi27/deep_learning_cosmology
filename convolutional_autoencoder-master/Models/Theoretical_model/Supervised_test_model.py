import h5py
import theano
import numpy as np

from supervised_cnn import *
from visualize import *
from guided_backprop import *

outputURL = '/global/homes/s/ssingh79/convolutional_autoencoder-master/output_files/'
foldername = 'Supervised_ss_1000_lr_5e-2'   ## Change this name to see others
modelfile = '/model.npz'

modelURL = outputURL + foldername + modelfile 

def main(outfolder = 'Tester'): 
    
    # Load test data
    hdf5file = outputURL + foldername + '/test_valid_data.h5'
    with h5py.File(hdf5file, 'r') as hf:
        X_test = hf['X_test'][:]
        Y_test = hf['Y_test'][:]
    
    # Create theano symbolic var for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    
    network, bottleneck_l = build_conv_ae(input_var) 

    # Load the trained model and set it back to the network! 
    with np.load(modelURL) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Test on X_test
    val_fn = theano.function([input_var, target_var], test_loss)
    
    # Create a theano funtion to get the prediction. 
    predict_fn = theano.function([input_var], test_prediction)
    
    # T-SNE plots 
    hidden_prediction = lasagne.layers.get_output(bottleneck_l, deterministic=True)
    hidden_fn = theano.function([input_var], hidden_prediction)
    
    #Let's check the Saliency map!
    def compile_saliency_function(bottleneck_l):
        inp = input_var
        outp = lasagne.layers.get_output(bottleneck_l, deterministic=True)
        max_outp = T.max(outp,axis=1)
        saliency = theano.grad(max_outp.sum(), wrt=inp)
        max_class = T.argmax(outp, axis=1)
        return theano.function([inp], [saliency, max_class])

    ################ Testing code #####################
    
    
    # After training, we compute and print the test error:
    test_err = 0
    #test_acc = 0
    test_batches = 0
    test_batch_size = X_test.shape[0]
    for batch in iterate_minibatches(X_test, Y_test, test_batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
 
    test_err_acc = []
    test_err_acc.append(test_err / test_batches)
    test_err_acc.append(test_acc / test_batches * 100)
    
    test_loss_fname = outputURL + outfolder + '/test_loss_accuracy.npz'
    np.savez(test_loss_fname, test_err_acc)
    
    ####################### Visualize code ######################### 
    
    # Save the Reconstruction images as hdf5 filessssss!!! 
    pred_images_fpath = outputURL + outfolder + '/test_reconstructed_maps.h5'
    sal_fpath  = outputURL + outfolder + '/test_saliency_maps.h5'
    
    #### RECONSTRUCTION ON TEST #########

    print("Running prediction function on Validation data")
    pred_images = predict_fn(X_test)
    # Reconstruction of images
    visualize_reconstruction(X_test, pred_images, outfolder)
    
    with h5py.File(pred_images_fpath,'w') as hf: 
        # X_train is the training set needed for unsupervised learning. 
        print("Creating hdf5 file for pred images and saving to ./output_files/.......")
        hf.create_dataset('norm_maps', data = X_test)
        hf.create_dataset('recon_maps', data = pred_images[0:X_test.shape[0]])
        
    ######## SALIENCY MAPS ############
    
    # Using Guided Backprop compute the non-linearities again!
    relu = lasagne.nonlinearities.rectify
    relu_layers = [layer for layer in lasagne.layers.get_all_layers(network)
                   if getattr(layer, 'nonlinearity', None) is relu]
    modded_relu = GuidedBackprop(relu)  # important: only instantiate this once!
    for layer in relu_layers:
        layer.nonlinearity = modded_relu
    
    # Visualizing Saliency map! 
    saliency_fn = compile_saliency_function(bottleneck_l) 
    X_sal, max_class = saliency_fn(X_test)
    visualize_saliency_map(X_test, X_sal, max_class, outfolder)
    
    ######## Visualize TSNE ##########
    print("Saving T-sne vector")
    bn_vector = hidden_fn(X_test)
    # Save the hidden layer output:
    np.savez(bn_fname, bn_vector)
    print("Visualizing t-sne")
    print("BN VECTOR ------", bn_vector.shape)
    Tsne_vector = visualize_tsne(bn_vector, Y_test, outfolder)
    # Embed Images into TSNE plot : 
    embed_img_into_tnse(X_test, Tsne_vector, outfolder)
    
    ## Save Saliency feature map! 
    with h5py.File(sal_fpath,'w') as hf:
        # X_train is the training set needed for unsupervised learning. 
        print("Creating hdf5 file for Salinecy maps saving to ./output_files/.......")
        hf.create_dataset('X_sal', data = X_sal[0:100])
        hf.create_dataset('max_class', data = max_class[0:100])

if __name__ == '__main__': 
    # If you want to specify command line arguments. 
    # Hyperparameter Optimizations here.
    if len(sys.argv) > 1:
        import argparse 
        parser = argparse.ArgumentParser(description='Command line options')
        parser.add_argument('--output', type=str, dest='outfolder')   
        args = parser.parse_args(sys.argv[1:])
        main(**{k:v for (k,v) in vars(args).items() if v is not None})
    else:
        main(outfolder = 'Test_Unsupervised_DCAE')        


