from unittest import TestCase
from petro_res_pack.parsing_utils import one_d_index_to_two, two_dim_index_to_one


class Test(TestCase):
    def test_one_d_index_to_two(self):
        self.assertEqual((0, 7), one_d_index_to_two(one_d=7, ny=8))
        self.assertEqual((1, 1), one_d_index_to_two(one_d=9, ny=8))
        self.assertEqual((0, 0), one_d_index_to_two(one_d=0, ny=1))
        self.assertEqual((2, 0), one_d_index_to_two(one_d=2, ny=1))

    def test_two_dim_index_to_one(self):
        self.assertEqual(0, two_dim_index_to_one(i=0, j=0, ny=4))
        self.assertEqual(1, two_dim_index_to_one(i=0, j=1, ny=4))
        self.assertEqual(4, two_dim_index_to_one(i=1, j=0, ny=4))
