from unittest import TestCase
import numpy as np
from petro_res_pack.sub_matrices_utils import get_sub_matrix


class TestGetSubMatrix(TestCase):
    x = np.array(range(3 * 4)).reshape(3, 4)

    def test_get_sub_matrix_1(self):
        self.assertTrue(np.array_equal(get_sub_matrix(self.x, 3, center=(0, 0), pad_value=-1),
                                       np.array([[-1, -1, -1],
                                                 [-1, 0, 1],
                                                 [-1, 4, 5]])
                                       )
                        )

    def test_get_sub_matrix_2(self):
        self.assertTrue(np.array_equal(get_sub_matrix(self.x, 5, center=(1, 1), pad_value=-1),
                                       np.array([[-1, -1, -1, -1, -1],
                                                 [-1, 0, 1, 2, 3],
                                                 [-1, 4, 5, 6, 7],
                                                 [-1, 8, 9, 10, 11],
                                                 [-1, -1, -1, -1, -1]])
                                       )
                        )

    def test_get_sub_matrix_3(self):
        self.assertTrue(np.array_equal(get_sub_matrix(self.x, 3, center=(1, 1), pad_value=-1),
                                       np.array([[0, 1, 2],
                                                 [4, 5, 6],
                                                 [8, 9, 10]])
                                       )
                        )

    def test_get_sub_matrix_4(self):
        try:
            get_sub_matrix(self.x, 3, center=(-1, 1), pad_value=-1)
            self.assertTrue(False)
        except IndexError:
            pass

    def test_get_sub_matrix_5(self):
        try:
            get_sub_matrix(self.x, 3, center=(1, 7), pad_value=-1)
            self.assertTrue(False)
        except IndexError:
            pass
