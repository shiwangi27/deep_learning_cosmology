__author__ = 'shiry'
import dictionary_utils as du
from create_SUN_category_dictionary import in_xor_out_door
import tables
import time
from random import seed, shuffle
import sys


def get_pure_doorness_categories(d):
    # returns the filtered dictionary containing only the categories that are indoor or outdoor but not both
    filtered_d = {k: v for (k,v) in d.items() if in_xor_out_door(d[k])}
    return filtered_d
    # TODO: test: filtered_d['/t/ticket_booth'] doesnt exist


def get_doorness_lists(d):
    in_list = [k for (k,v) in get_pure_doorness_categories(d).items() if v['indoor']]
    out_list = [k for (k,v) in get_pure_doorness_categories(d).items() if not v['indoor']]
    return in_list, out_list


def save_list(lst, filename):
    f = open(filename, "w")
    f.write("content = " + str(lst))
    f.close()

def file_list_for_categories(categories, h5_handle):
    file_list = []
    for category in categories:
        try:
            h5_handle.root._f_get_child(category)
        except tables.NoSuchNodeError:
            continue
        catGroup = h5_handle.root._f_get_child(category)
        for node in catGroup._v_children.iterkeys():
            file_list.append(catGroup._v_pathname + "/" + node)
    return file_list

if __name__ == "__main__":
    # load the dictionary that associates each category with an "indoor" vs "outdoor" specification
    # but filter it so it only includes categories that are not double-marked as both indoor and outdoor
    # and return the lists of the indoor vs the outdoor categories
    (indoor_list, outdoor_list) = get_doorness_lists(du.load_dictionary('SUN908_inoutdoor_dictionary'))
    # get the list of images for each "doorness" condition.

    # load the HDF5 file that was created using the pytables library
    # TODO put this back?
    with tables.open_file('/clusterfs/cortex/scratch/shiry/places32.h5', 'r') as h:
        # every array corresponding to one image can be accessed like so:
        # h.root.a.abbey.<IMAGE NAME>[:]

        seed(1);
        #t = time.time()
        file_list = file_list_for_categories(outdoor_list, h)

        #print(time.time() - t)
        shuffle(file_list)

        #print(len(file_list))
        save_list(file_list,'outdoor_file_list.py')
    
        file_list = file_list_for_categories(indoor_list, h)

        #print(time.time() - t)
        # shuffle the lists to get a random walk over the images in each category
        shuffle(file_list)
        save_list(file_list,'indoor_file_list.py')
	
	    # to get the image-data from the hdf5 file:
        #h.root._f_get_child(indoor_file_list[0])[:]




