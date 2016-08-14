import os
import h5py
import numpy as np
from scipy import misc 

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

outputURL = '/global/homes/s/ssingh79/convolutional_autoencoder-master/output_files/'
        
        
def embed_img_into_tnse(X_train, Tsne_vector, foldername):
    #print("Location --------------------", Tsne_vector)
    print("Tsne vector shape ----------- ", Tsne_vector.shape)
    
    # Take the embedding 2d locations from tsne plot! 
    Tsne_vector = np.subtract(Tsne_vector, np.min(Tsne_vector))
    Tsne_vector = np.divide(Tsne_vector, np.max(Tsne_vector))
    
    loc1 = Tsne_vector[:,0]
    loc2 = Tsne_vector[:,1]
    
    # Create an embedding Image G of Size=(S,S) to paste every image of size=(s,s)
    S = 2000
    G = np.ones((S,S)).astype('float')    
    s = 50 
    #plt.imshow(G, cmap='Greys_r')
    
    # Number of Images in the sample! 
    N = X_train.shape[0]
    
    for i in range(0,N):
        if(np.mod(i,100)==0):
            print('Done -- ', i, '/', N)
            
        # Location calculation 
        embed_x = np.ceil(loc1[i]*(S-s)+1)
        embed_y = np.ceil(loc2[i]*(S-s)+1)
        embed_x = embed_x - np.mod(embed_x+1,s)+1
        embed_y = embed_y - np.mod(embed_y+1,s)+1
    
        # Check if spot is already filled
        if(G[embed_x,embed_y]!=1.0):
            continue
        
        # Read image files from the training data 
        I = X_train[i][0]
        I = np.resize(I, (s,s))
        
        #I = mpimg.imread('/global/cscratch1/sd/ssingh79/TSNE_images/'+ str(i)+'.png')
        #I = np.resize(I,(s,s))
        #img = Image.open('/global/cscratch1/sd/ssingh79/TSNE_images/'+ str(i)+'.PNG')
        #I = img.thumbnail((s, s), Image.ANTIALIAS)
        
        G[embed_x:embed_x+s, embed_y:embed_y+s] = I 
    #name='hot'
    plt.imshow(G) 
    plt.colorbar()
    embedTsne_fpath = os.path.join(outputURL, foldername + '/TSNE_embedded.png')
    plt.savefig(embedTsne_fpath)
    plt.close()
    
    
    
    