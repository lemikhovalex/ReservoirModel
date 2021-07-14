from unittest import TestCase
from petro_res_pack.parsing_utils import one_d_index_to_two, two_dim_index_to_one


class TestTwoDimIndexToOne(TestCase):
    def test_two_dim_index_to_one1(self):
        self.assertEqual(0, two_dim_index_to_one(i=0, j=0, ny=4))

    def test_two_dim_index_to_one2(self):
        self.assertEqual(1, two_dim_index_to_one(i=0, j=1, ny=4))

    def test_two_dim_index_to_one3(self):
        self.assertEqual(4, two_dim_index_to_one(i=1, j=0, ny=4))


class TestOneDIndexToTwo(TestCase):
    def test_one_d_index_to_two1(self):
        self.assertEqual((0, 7), one_d_index_to_two(one_d=7, ny=8))

    def test_one_d_index_to_two2(self):
        self.assertEqual((1, 1), one_d_index_to_two(one_d=9, ny=8))

    def test_one_d_index_to_two3(self):
        self.assertEqual((0, 0), one_d_index_to_two(one_d=0, ny=1))

    def test_one_d_index_to_two4(self):
        self.assertEqual((2, 0), one_d_index_to_two(one_d=2, ny=1))
