from unittest import TestCase
from create_HDF5 import save_hdf5
import tables
import numpy as np

__author__ = 'shiry'


class TestSave_hdf5(TestCase):
    def test_save_hdf5(self):
        # save an HDF5 file equivalent to the given testdir hierarchy
        save_hdf5('testdir_hdf5', 'test_hdf5_output.h5', 'vec')
        # load the written HDF5 file
        h = tables.open_file('test_hdf5_output.h5', 'r')
        # assert that one of the Leaf arrays (in a file called place) is in the right place (/a/category_in_a/test.mat)
        # and contains the correct numerical data stored in a numpy array
        self.assertTrue(np.array_equal(h.root.a.category_in_a.test[:], np.array([1, 2, 3, 4, 5])))
        h.close()