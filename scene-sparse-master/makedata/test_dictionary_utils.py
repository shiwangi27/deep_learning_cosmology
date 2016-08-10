from unittest import TestCase
import dictionary_utils as du
from create_SUN_category_dictionary import in_xor_out_door
from create_SUN_category_dictionary import natural_xor_man_made

__author__ = 'shiry'


class TestDictionary_utils(TestCase):
    def test_load_dictionary(self):
        d = du.load_dictionary('SUN908_inoutdoor_dictionary')
        self.assertTrue(d['/a/art_studio']['indoor'])
        # test for categories that are not marked as indoor or outdoor
        self.assertFalse(any(d['/a/archaelogical_excavation'].values()))
        # test for categories that are marked as both indoor and outdoor
        self.assertFalse(in_xor_out_door(d['/t/ticket_booth']))
        # test for categories that are marked as both outdoor natural and outdoor man-made
        self.assertFalse(natural_xor_man_made(d['/a/aqueduct']))

