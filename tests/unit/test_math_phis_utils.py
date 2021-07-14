from unittest import TestCase
import numpy as np
from petro_res_pack.math_phis_utils import one_zero_swap, get_ax_update, chose_sat_for_upd
from petro_res_pack.res_state import ResState
from petro_res_pack.properties import Properties


class TestOneZeroSwap(TestCase):
    def test_one_zero_swap1(self):
        self.assertEqual(1, one_zero_swap(x=0))

    def test_one_zero_swap0(self):
        self.assertEqual(0, one_zero_swap(x=1))


class TestGetAxUpdate(TestCase):
    nx = 2
    ny = 3
    prop = Properties(nx=nx, ny=ny)
    reservoir_state = ResState(values=np.array(range(nx * ny)), bound_value=-1, prop=prop)

    def test_get_ax_update1(self):
        self.assertTrue(np.array_equal(np.array([-1, 0, 1, 2, -1, 3, 4, 5, -1]),
                                       get_ax_update(state=self.reservoir_state, prop=self.prop, axis=0))
                        )

    def test_get_ax_update2(self):
        self.assertTrue(np.array_equal(np.array([-1, 0, 3, -1, 1, 4, -1, 2, 5, -1]),
                                       get_ax_update(state=self.reservoir_state, prop=self.prop, axis=1))
                        )


class TestChoseSatForUpd(TestCase):
    def test_chose_sat_for_upd1(self):
        expected = np.array([0, 3, 3, 1, 4, 4, 2, 5, 5])
        get_value = chose_sat_for_upd(p=np.array([-1, 0, 3, -1, 1, 4, -1, 2, 5, -1]),
                                      s=np.array([-1, 0, 3, -1, 1, 4, -1, 2, 5, -1])
                                      )
        self.assertTrue(np.array_equal(expected, get_value), msg=f'\nexp={expected}\nget={get_value}')

    def test_chose_sat_for_upd2(self):
        expected = np.array([-1, 0, 1, 2])
        get_value = chose_sat_for_upd(p=np.array([3, 2, 1, 0, -1]),
                                      s=np.array([-1, 0, 1, 2, 3])
                                      )
        self.assertTrue(np.array_equal(expected, get_value), msg=f'\nexp={expected}\nget={get_value}')

    def test_chose_sat_for_upd3(self):
        expected = np.array([2, 2, 4, 4, 6, 7, 7, 8])
        get_value = chose_sat_for_upd(p=np.array([4,  5,  1,  8, -1, 17, 25,  9,  0]),
                                      s=np.array([1,  2,  3,  4,  5,  6,  7,  8,  9])
                                      )
        self.assertTrue(np.array_equal(expected, get_value), msg=f'\nexp={expected}\nget={get_value}')
