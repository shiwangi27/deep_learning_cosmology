import importlib
import tables


def load_list(filename_no_extension):
    """
    :type filename_no_extension: A filename string without the .py extension
    """
    print('loading file ' + filename_no_extension)
    module = importlib.import_module(filename_no_extension)
    assert isinstance(module, object)
    assert isinstance(module.content, list)
    return module.content


if __name__ == "__main__":
    indoor_file_list = load_list('indoor_file_list')
    outdoor_file_list = load_list('outdoor_file_list')

    # to get the image-data from the hdf5 file:
    with tables.open_file('/clusterfs/cortex/scratch/shiry/places32.h5', 'r') as h:
        # you can go over each of the lists and use their elements as keys into the hdf5 file like so:
        image_vector = h.root._f_get_child(indoor_file_list[0])[:]
        print(image_vector)
