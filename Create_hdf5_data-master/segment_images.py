#This is the code for resizing the image data and experimenting with the training set. 

import numpy as np 
import h5py 
import os 
import sys

def load_data():

    # Load the dataset
    
    # Here we give different data sets for the autoencoders
    dataurl = '/global/cscratch1/sd/ssingh79/si85_data/'
    #hdf5file = 'conv_z02.h5'
    hdf5file = 'si85_z02.h5'
    filepath = os.path.join(dataurl, hdf5file)
    
    print("Calling ", hdf5file, "......")
    # Call the load_data method to get back the Final training set. 
    dataset = filepath
    
    with h5py.File(dataset,'r') as hf: 
        #train_set = hf['X_train'][0:1000,0:65536] 
        train_set = hf['X_train'][0:1000,:]
        print("Printing Train set ", train_set) 
        print("X_train shape ", train_set.shape)
        
        return train_set
'''    
def normalizer():
    # This function normalizes the dataset by rows using the linalg.norm 
    # numpy module. The formula is Xi / sqrt(( Xi - mu )**2)
    # The values are in the range of [-1,1]. 
    
    X_train = load_data()
    X_norm = np.linalg.norm(X_train, axis=0)
    X_train = X_train/X_norm[np.newaxis,:]
    print("Normalized : ", X_train)
    print("Mean : ", np.mean( X_train))
    print("min : ", np.min(X_train))
    print("max : ", np.max(X_train))
    
    #image = np.reshape(X_train[0,:],(1024,1024))
    
    return X_train
'''    
    
def seg_images(size, start_index, end_index):
    #Segments the images into a given size    
    train_x = load_data()
    
    #Initialize segmeted images
    segmented_images = np.zeros((1,size*size))
    
    for i in range(start_index, end_index):
        
        #Take each 1024x1024 size image to segment.
        temp_x = np.reshape(train_x[i,:],(1024,1024))
        
        #Carve 64 Image patches from each data sample.
        #Total number of samples becomes 64 x 1000 = 64,000.
        
        for r in range(8):
            row_start = size*r
            column_start = 0
            for c in range(8):
                column_start = size*c
                
                #temp_image holds 128x128 size image in every iteration. 
                temp_image = temp_x[row_start:row_start+size,column_start:column_start+size]
                
                #print("2D Slice : ", temp_image)
                print("Starting row :", row_start)
                print("Starting column : ", column_start)
        
                #Reshape back each of the segmented images.
                temp_image = np.reshape(temp_image, (1,size*size))
        
                #Stack all the reshaped images back for creating a h5py file. 
                segmented_images = np.vstack((segmented_images, temp_image))
                
            print("Shape of the segmented_images so far : ",segmented_images.shape)
        
        print("Exiting iteration ---------------------------------- ", i)
    
    #Delete the first row initialization with 0 done before. 
    segmented_images = np.delete(segmented_images,(0),axis=0)

    print(segmented_images)
    print(segmented_images.shape)
    
    # See the image lol.  
    print_seg = np.reshape(segmented_images[0,:],(size,size))
    #%matplotlib inline
    #plt.imshow(print_seg, interpolation = 'None')
    #plt.colorbar()
    
    return segmented_images
        
def save_dataset(batch_num=1, start_index=0, end_index=250):
    
    #Specify size you want to carve out here. 
    size = 128 
    
    segmented_data = seg_images(size, start_index, end_index)
    
    # Store the resized images to a hdf5 file. 
    
    dataURL = '/global/cscratch1/sd/ssingh79/si85_data/'
    filename = 'segmented16k_data_'
    
    datapath = os.path.join( dataURL, filename + str(batch_num) + '.h5')
    print("The batch path is ", datapath)
    
    with h5py.File(datapath,'w') as hf:
        # X_train is the training set needed for unsupervised learning. 
       
        print("Creating h5py data files and saving to ./si85_data/segment16k_data.h5 .........")
        hf.create_dataset('X_train64k', data = segmented_data[:,:]) 

if __name__ == '__main__':
    if len(sys.argv) > 1:
        batch_n = int(sys.argv[1])
        start_idx = int(sys.argv[2])
        end_idx = int(sys.argv[3])
        save_dataset(batch_num=batch_n, start_index=start_idx, end_index=end_idx)
        
         