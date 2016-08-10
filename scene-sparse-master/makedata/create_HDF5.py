__author__ = 'shiry'
import argparse
import tables # pytables
import os
import scipy.io


def save_hdf5(data_dir, output_filename, image_vector_varname):
    # create the output HDF5 file and open it for writing using pytables
    fh = tables.open_file(output_filename, 'w')
    # go over the files in the image data directory
    if not data_dir.endswith('/'):
        data_dir += '/'

    filters = tables.Filters(complevel=5, complib='zlib')

    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        # get rid of the input data_dir in the HDF5 path
        group_path = dirpath[len(data_dir)-1:]
        # create a pytables Group (equivalent to a directory) for each sub directory in the hierarchy
        for dirname in dirnames:
            fh.create_group(group_path, dirname)
        # create a pytable Leaf (equivalent to a file) of type Array for each image data file in the hierarchy
        for filename in filenames:
            print(filename)
            # read the current image's mat file into a numpy array
            image_array = scipy.io.loadmat(os.path.join(dirpath, filename), squeeze_me=True)[image_vector_varname]
            # stick the data in a compression-enabled array
            fh.create_carray(group_path, filename.split('.')[0], obj=image_array, filters=filters)
    fh.flush()
    fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert the image data hierarchy into an HDF5 file with pytables Group hierarchy and save it.')
    parser.add_argument('data_dir', type=str, help='the input data directory containing the image vectors')
    parser.add_argument('--output_filename', dest='output_filename', type=str,
                        help='the output HDF5 filename -- should be of the form X.h5', default='/clusterfs/cortex/scratch/shiry/places32.h5')
    parser.add_argument('--image_vector_varname', dest='image_vector_varname', type=str,
                        help='the name of the vector variable in each image-vector mat file', default='I_tiny')
    args = parser.parse_args()

    save_hdf5(args.data_dir, args.output_filename, args.image_vector_varname)


