#This code just generates the mean images for indoor,outdoor and combined
#This is later used by the code but can also be used for analysis


import importlib
import tables
import numpy as np
import scipy.io as scio

def load_list(filename_no_extension):
    """
    :type filename_no_extension: A filename string without the .py extension
    """
    print('loading file ' + filename_no_extension)
    module = importlib.import_module(filename_no_extension)
    assert isinstance(module, object)
    assert isinstance(module.content, list)
    return module.content

if __name__=='__main__':
    print('Loading File lists')
    indoor_file_list=load_list('indoor_file_list')
    outdoor_file_list=load_list('outdoor_file_list')

    #Calculate Indoor File average
    idx=0
    indoor_image=np.zeros([32*32,1])
    outdoor_image=np.zeros([32*32,1])
    mean_images={'indoor_mean':indoor_image,'outdoor_mean':outdoor_image,'dataset_mean':outdoor_image}
    try:
        h= tables.open_file('/media/mudigonda/Gondor/Data/scene-sparse/places32.h5','r') 
    except:
        print('Could not get file handle. Aborting')
    for ii in range(len(indoor_file_list)):
        if np.mod(ii,1000)==0:
            print('The value of indoor images processed is -- ii--',ii)
        try:
            indoor_image_tmp = h.root._f_get_child(indoor_file_list[ii])[:]
        except:
            print('Could not extract indoor image ',indoor_file_list[ii])
        indoor_image = indoor_image + indoor_image_tmp

    for jj in range(len(outdoor_file_list)):
        if np.mod(jj,1000)==0:
            print('The value of outdoor images processed is --jj--',jj)
        try:
            outdoor_image_tmp = h.root._f_get_child(outdoor_file_list[jj])[:]
        except:
            print('Could not extract outdoor image ',outdoor_file_list[ii])
        outdoor_image = outdoor_image + outdoor_image_tmp
    
    mean_images['indoor_mean'] = indoor_image/len(indoor_file_list)
    mean_images['outdoor_mean'] = outdoor_image/len(outdoor_file_list)
    mean_images['dataset_mean'] = (indoor_image+outdoor_image)/(len(indoor_file_list)+len(outdoor_file_list))

    scio.savemat('mean_images.mat',mean_images)
    h.close()

