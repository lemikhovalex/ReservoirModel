from unittest import TestCase
import numpy as np

from petro_res_pack.res_state import ResState
from petro_res_pack.properties import Properties


class TestResState(TestCase):
    prop = Properties()
    prop.nx = 3
    prop.ny = 4
    state = ResState(values=np.array(range(prop.nx * prop.ny)), bound_value=-1, prop=prop)

    def test_get_item_1(self):
        state = ResState(values=np.array(range(self.prop.nx * self.prop.ny)), bound_value=-1, prop=self.prop)
        self.assertEqual(state[0, 0], 0)

    def test_get_item_2(self):
        try:
            _ = ResState(values=np.array(range(self.prop.nx - 1 * self.prop.ny)), bound_value=-1, prop=self.prop)
        except IndexError:
            pass

    def test_get_item_3(self):
        state = ResState(values=np.array(range(self.prop.nx * self.prop.ny)), bound_value=-1, prop=self.prop)
        self.assertEqual(state[0, 2], 2)

    def test_get_item_4(self):
        state = ResState(values=np.array(range(self.prop.nx * self.prop.ny)), bound_value=-1, prop=self.prop)
        self.assertEqual(state[1, 1], 5)

    def test_get_item_5(self):
        self.assertEqual(self.state[-0.5, 1], -1)

    def test_get_item_6(self):
        self.assertEqual(self.state[1, -0.5], -1)

    def test_get_item_7(self):
        try:
            _ = self.state[1, self.prop.ny + 1]
            self.assertTrue(True, msg='')
        except IndexError:
            pass
