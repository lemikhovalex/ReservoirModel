from abc import ABC

import numpy as np
import scipy
import gym
import utils as u
import seaborn as sns
from IPython import display
import matplotlib.pyplot as plt

from reservoir import ResState, get_j_matrix, get_q_bound, get_lapl_one_ph_np, get_r_ref
from res_properties import Properties


s_star = 0.356

class petro_env(gym.Env):
    def __init__(self, p, s_o: ResState, s_w: ResState, prop: Properties, pos_r: dict, delta_p_well: float, max_time=90):
        self.max_time = max_time
        self.p_0 = p
        self.s_o_0 = s_o
        self.s_w_0 = s_w
        self.prop_0 = prop
        self.delta_p_well = delta_p_well

        self.times = []
        self.p = p
        self.s_o = s_o
        self.s_w = s_w
        self.prop = prop
        self.pos_r = pos_r

        self.j_o = np.zeros((prop.nx * prop.ny, 1))
        self.j_w = np.zeros((prop.nx * prop.ny, 1))

        self.q_bound_w = np.zeros((prop.nx * prop.ny, 1))
        self.q_bound_o = np.zeros((prop.nx * prop.ny, 1))

        self.price = {'w': 5, 'o': 40}

        # self.delta_p_vec = np.ones((prop.nx * prop.ny, 1)) * delta_p_well
        # '''
        self.delta_p_vec = np.zeros((prop.nx * prop.ny, 1))
        for pos in pos_r:
            self.delta_p_vec[u.two_dim_index_to_one(pos[0], pos[1], ny=prop.ny), 0] = delta_p_well
        # '''

        self.nxny_ones = np.ones((prop.nx * prop.ny, 1))
        self.nxny_eye = np.eye(prop.nx * prop.ny)
        self.t = 0
        self.lapl_o = None
        self.dt_comp_sat = None
        self.lapl_w = None
        self.openity = np.zeros(self.prop.nx * self.prop.ny)
        self.s_star = 0
        self.set_s_star()

    def set_s_star(self):
        min_d = 100
        _s_star = 1
        for ss in np.linspace(0, 1, 20000):
            w_ben = self.price['w'] * self.prop.k_rel_w(1 - ss) / self.prop.mu['w']
            o_ben = self.price['o'] * self.prop.k_rel_o(ss) / self.prop.mu['o']
            d = abs(w_ben - o_ben)
            if d < min_d:
                _s_star = ss
                min_d = d
        self.s_star = _s_star

    def prepro_s(self, s_o: ResState) -> np.ndarray:
        out = s_o.v - self.s_star
        out /= [0.5 - 0.2]
        out[out > 0] += 0.1
        out[out < 0] -= 0.1
        return out

    def get_observation(self, s_o: ResState, p: ResState, prop: Properties):
        s_o_sc = self.prepro_s(s_o)
        return np.concatenate((s_o_sc, p.v / p.bound_v / 10.0), axis=None)

    def step(self, action: np.ndarray = None) -> list:
        """
        action is a np.ndarray with
        """
        if action is not None:
            assert len(self.pos_r) == len(action)  # wanna same wells

        self.dt_comp_sat = self.s_o.v * self.prop.c['o'] + self.s_w.v * self.prop.c['w']
        self.dt_comp_sat += self.nxny_ones * self.prop.c['r']
        self.dt_comp_sat *= self.prop.dx * self.prop.dy * self.prop.d
        # do matrixes for flow estimation
        get_j_matrix(p=self.p, s=self.s_o, pos_r=self.pos_r, ph='o', prop=self.prop, j_matr=self.j_o)
        get_j_matrix(p=self.p, s=self.s_w, pos_r=self.pos_r, ph='w', prop=self.prop, j_matr=self.j_w)
        # wells are open not full-wide
        self.openity = np.ones((self.prop.nx * self.prop.ny, 1))
        if action is not None:
            for _i, well in enumerate(self.pos_r):
                self.openity[u.two_dim_index_to_one(well[0], well[1], self.prop.ny), 0] = action[_i]
            self.j_o *= self.openity
            self.j_w *= self.openity
        # now
        self.lapl_w, si_w = get_lapl_one_ph_np(p=self.p, s=self.s_w, ph='w', prop=self.prop)
        self.lapl_o, si_o = get_lapl_one_ph_np(p=self.p, s=self.s_o, ph='o', prop=self.prop)

        get_q_bound(self.p, self.s_w, 'w', self.prop, q_b=self.q_bound_w)
        get_q_bound(self.p, self.s_o, 'o', self.prop, q_b=self.q_bound_o)
        # self.prop.dt = 0.1 * 0.5 * self.prop.phi * self.dt_comp_sat.min() / (si_o + si_w)
        # set dt accoarding Courant
        self.prop.dt = 0.1 * self.prop.phi * self.dt_comp_sat.min() / (si_o + si_w)
        # matrix for implicit pressure
        a = self.prop.phi * scipy.sparse.diags(diagonals=[self.dt_comp_sat.reshape(-1)],
                                               offsets=[0])
        # a = self.nxny_eye *  self.prop.phi * self.dt_comp_sat
        a = a - (self.lapl_w + self.lapl_o) * self.prop.dt
        # right hand state for ax = b
        b = self.prop.phi * self.dt_comp_sat * self.p.v + self.q_bound_w * self.prop.dt + self.q_bound_o * self.prop.dt
        b += (self.j_o * self.prop.b['o'] + self.j_w * self.prop.b['w']) * self.delta_p_vec * self.prop.dt
        # solve p
        p_new = scipy.sparse.linalg.spsolve(a, b).reshape((-1, 1))
        # upd time stamp

        self.t += self.prop.dt / (60. * 60 * 24)

        a = self.nxny_ones + (self.prop.c['r'] + self.prop.c['o']) * (p_new - self.p.v)
        a *= self.prop.dx * self.prop.dy * self.prop.d * self.prop.phi

        b = self.prop.phi * self.prop.dx * self.prop.dy * self.prop.d * self.s_o.v
        b += self.prop.dt * (self.lapl_o.dot(p_new) + self.q_bound_o + self.j_o * self.prop.b['o'] * self.delta_p_vec)
        # upd target values
        self.s_o = ResState((b / a), self.s_o.bound_v, self.prop)
        self.s_w = ResState(self.nxny_ones - self.s_o.v, self.s_w.bound_v, self.prop)
        self.p = ResState(p_new, self.prop.p_0, self.prop)

        obs = self.get_observation(s_o=self.s_o, p=self.p, prop=self.prop)

        reward = self.get_reward()

        return [obs, reward, self.t > self.max_time, {}]

    def get_q(self, ph):
        out = None
        if ph == 'o':
            out = ((-1) * self.j_o * self.delta_p_vec).reshape((self.prop.nx, self.prop.ny))
        elif ph == 'w':
            out = ((-1) * self.j_w * self.delta_p_vec).reshape((self.prop.nx, self.prop.ny))
        return out * self.openity.reshape((self.prop.nx, self.prop.ny))

    def get_reward(self):
        q_o = self.get_q('o')
        q_w = self.get_q('w')

        return (self.price['o'] * q_o.sum() - self.price['w'] * q_w.sum()) * self.prop.dt

    def reset(self):
        self.p.v = np.ones((self.prop.nx * self.prop.ny, 1)) * self.p.bound_v
        self.s_o.v = np.ones((self.prop.nx * self.prop.ny, 1)) * self.prop.s_0['o']
        self.s_w.v = np.ones((self.prop.nx * self.prop.ny, 1)) * self.prop.s_0['w']
        self.pos_r = self.pos_r

        self.j_o = np.zeros((self.prop.nx * self.prop.ny, 1))
        self.j_w = np.zeros((self.prop.nx * self.prop.ny, 1))

        self.q_bound_w = np.zeros((self.prop.nx * self.prop.ny, 1))
        self.q_bound_o = np.zeros((self.prop.nx * self.prop.ny, 1))

        # self.delta_p_vec = np.ones((prop.nx * prop.ny, 1)) * delta_p_well
        # '''
        self.delta_p_vec = np.zeros((self.prop.nx * self.prop.ny, 1))
        for pos in self.pos_r:
            self.delta_p_vec[u.two_dim_index_to_one(pos[0], pos[1], ny=self.prop.ny), 0] = self.delta_p_well
        # '''

        self.nxny_ones = np.ones((self.prop.nx * self.prop.ny, 1))
        self.nxny_eye = np.eye(self.prop.nx * self.prop.ny)
        self.t = 0
        self.lapl_o = None
        self.dt_comp_sat = None
        self.lapl_w = None
        self.openity = np.zeros(self.prop.nx * self.prop.ny)
        obs = self.get_observation(s_o=self.s_o, p=self.p, prop=self.prop)
        return obs
