from unittest import TestCase
from petro_res_pack.properties import Properties
import numpy as np


class TestGetSWn(TestCase):
    prop = Properties()

    def test_get_s_wn_1(self):
        _ = self.prop._get_s_wn(s_w=np.array([0, 1, 0.5]))

    def test_get_s_wn_2(self):
        _ = self.prop._get_s_wn(s_w=1)

    def test_get_s_wn_3(self):
        _ = self.prop._get_s_wn(s_w=1)


class TestKRelW(TestCase):
    prop = Properties()

    def test_k_rel_w_1(self):
        _ = self.prop._k_rel_w(s_w=np.array([0, 1, 0.5]))

    def test_k_rel_w_2(self):
        _ = self.prop._k_rel_w(s_w=1)

    def test_k_rel_w_3(self):
        _ = self.prop._k_rel_w(s_w=1)


class TestKRelO(TestCase):
    prop = Properties()

    def test_k_rel_o_1(self):
        _ = self.prop._k_rel_o(s_o=np.array([0, 1, 0.5]))

    def test_k_rel_o_2(self):
        _ = self.prop._k_rel_o(s_o=1)

    def test_k_rel_o_3(self):
        _ = self.prop._k_rel_o(s_o=1)


class TestKRelPh(TestCase):
    prop = Properties()

    def test_k_rel_by_ph_1(self):
        _ = self.prop.k_rel_by_ph(s=np.array([0, 1, 0.5]), ph='o')

    def test_k_rel_by_ph_2(self):
        _ = self.prop.k_rel_by_ph(s=0, ph='w')

    def test_k_rel_by_ph_3(self):
        _ = self.prop.k_rel_by_ph(s=1, ph='o')

    def test_k_rel_by_ph_4(self):
        try:
            _ = self.prop.k_rel_by_ph(s=1, ph='ggg')
            self.assertTrue(False, msg='Passed wrong ard')
        except ValueError:
            pass


class TestKRelPhLocalPressureDecision(TestCase):
    prop = Properties()

    def test_k_rel_ph_local_pressure_decision(self):
        _ = self.prop.k_rel_ph_local_pressure_decision(s_1=0.5, s_2=0.5, p_1=4, p_2=5, ph='o')

    def test_k_rel_ph_local_pressure_decision_2(self):
        _ = self.prop.k_rel_ph_local_pressure_decision(s_1=0.5, s_2=0.5, p_1=24, p_2=5, ph='w')

    def test_k_rel_ph_local_pressure_decision_3(self):
        try:
            _ = self.prop.k_rel_ph_local_pressure_decision(s_1=0.5, s_2=0.5, p_1=24, p_2=5, ph='ww')
            self.assertTrue(False, msg='Passed wrong arg as ph')
        except ValueError:
            pass
