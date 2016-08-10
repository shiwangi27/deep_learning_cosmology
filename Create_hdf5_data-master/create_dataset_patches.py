import numpy as np
import random
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

imsize = 1024
border = 4 
sz = 90 
patch_dim = 90*90
# This fucker takes 128x128 images and gives you random patches of 32x32 !!!

print("Loading data ..............")
with h5py.File('/global/homes/s/ssingh79/data/conv_z02.h5','r') as hf:
    IMAGES = hf['X_train'][:,:]
    print(IMAGES.shape)

print("Let's do some transformation !!!")

#IMAGES = IMAGES.transpose()
#print(IMAGES.shape)
IMAGES = IMAGES.reshape(-1,1024,1024)
print(IMAGES.shape)

#images = np.random.random((128,128,1000))
#print(images.shape)

batch = 100000
data = np.zeros((batch, patch_dim))

for i in range(batch):
    print("Taking patch for batch -- iteration : ", i)
    imi = np.ceil(1000*random.uniform(0,1))
    print("Image number : ", imi)
    r = border + np.ceil((imsize-sz-2*border) * random.uniform(0,1))
    c = border + np.ceil((imsize-sz-2*border) * random.uniform(0,1))
    data[i,:] = np.reshape(IMAGES[imi-1, r:r+sz, c:c+sz],patch_dim,1)
    
print("The 1M dataset is : ", data)
print(data.shape)

with h5py.File('/global/cscratch1/sd/ssingh79/data/train_samples_1M_p90.h5','w') as hf:
    print("Creating new dataset .............")
    hf.create_dataset('X_train', data=data)

with h5py.File('/global/cscratch1/sd/ssingh79/data/train_samples_1M_p90.h5','r') as hf:
    print("Saving 100 images randomly from the samples..........")
    for i in range(0,100): 
        idx = np.ceil(batch*random.uniform(0,1)) 
        image = hf['X_train'][idx,:] 
        image = image.reshape(90,90) 
        plt.imshow(image) 
        plt.savefig('/global/homes/s/ssingh79/Image_Patches_90/train_image_' + str(i))
        plt.colorbar()
        plt.close()
        
        

    