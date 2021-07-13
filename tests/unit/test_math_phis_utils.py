from unittest import TestCase
import numpy as np
from petro_res_pack.math_phis_utils import one_zero_swap, get_ax_update
from petro_res_pack.res_state import ResState
from petro_res_pack.properties import Properties


class TestMathPhisUtils(TestCase):
    nx = 2
    ny = 3
    prop = Properties(nx=nx, ny=ny)
    reservoir_state = ResState(values=np.array(range(nx * ny)), bound_value=-1, prop=prop)

    def test_one_zero_swap(self):
        self.assertEqual(0, one_zero_swap(x=1))
        self.assertEqual(1, one_zero_swap(x=0))

    def test_get_ax_update(self):
        self.assertTrue(np.array_equal(np.array([-1, 0, 1, 2, -1, 3, 4, 5, -1]),
                                       get_ax_update(state=self.reservoir_state, prop=self.prop, axis=0))
                        )

        self.assertTrue(np.array_equal(np.array([-1, 0, 3, -1, 1, 4, -1, 2, 5, -1]),
                                       get_ax_update(state=self.reservoir_state, prop=self.prop, axis=1))
                        )
