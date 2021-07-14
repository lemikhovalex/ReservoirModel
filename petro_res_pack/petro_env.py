import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_lin_alg
from gym import Env
from gym.envs.classic_control.rendering import SimpleImageViewer

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns

import pandas as pd

from .properties import Properties
from .res_state import ResState
from .parsing_utils import two_dim_index_to_one
from .sub_matices_utils import get_sub_matrix
from .math_phis_utils import get_laplace_one_ph


def get_q_bound(p: ResState, s, ph, prop: Properties, q_b):
    q_b *= 0
    for row in range(prop.nx):
        # (row, -0.5)
        k_r = prop.k_rel_ph_local_pressure_decision(s_1=s[row, -0.5], s_2=s[row, 0],
                                                    p_1=p[row, -0.5], p_2=p[row, 0],
                                                    ph=ph)

        dia_ = two_dim_index_to_one(i=row, j=0, ny=prop.ny)
        q_b[dia_, 0] += prop.k * k_r / prop.dx * p[row, -0.5] * prop.d * prop.dy / prop.mu[ph]
        # (row, ny-0.5)
        k_r = prop.k_rel_ph_local_pressure_decision(s_1=s[row, prop.ny - 1], s_2=s[row, prop.ny - 0.5],
                                                    p_1=p[row, prop.ny - 1], p_2=p[row, prop.ny - 0.5],
                                                    ph=ph)
        dia_ = two_dim_index_to_one(i=row, j=prop.ny - 1, ny=prop.ny)
        q_b[dia_, 0] += prop.k * k_r / prop.dx * p[row, prop.ny - 0.5] * prop.d * prop.dy / prop.mu[ph]

    for col in range(prop.ny):
        # (-0.5, col)
        k_r = prop.k_rel_ph_local_pressure_decision(s_1=s[-0.5, col], s_2=s[0, col],
                                                    p_1=p[-0.5, col], p_2=p[0, col],
                                                    ph=ph)
        dia_ = two_dim_index_to_one(i=0, j=col, ny=prop.ny)
        q_b[dia_, 0] += prop.k * k_r / prop.dx * p[prop.nx - 0.5, col] * prop.d * prop.dy / prop.mu[ph]
        # (nx-0.5, col)
        k_r = prop.k_rel_ph_local_pressure_decision(s_1=s[prop.nx - 1, col], s_2=s[prop.nx - 0.5, col],
                                                    p_1=p[prop.nx - 1, col], p_2=p[prop.nx - 0.5, col],
                                                    ph=ph)
        dia_ = two_dim_index_to_one(i=prop.nx - 1, j=col, ny=prop.ny)
        q_b[dia_, 0] += prop.k * k_r / prop.dx * p[prop.nx - 0.5, col] * prop.d * prop.dy / prop.mu[ph]
        # corners to zero
    # return q_b.reshape((prop.nx * prop.ny, 1))


def get_r_ref(prop: Properties):
    return 1 / (1 / prop.dx + np.pi / prop.d)


def get_j_matrix(s, p, pos_r, ph, prop: Properties, j_matrix):
    r_ref = get_r_ref(prop)
    for pos in pos_r:
        dia_pos = two_dim_index_to_one(i=pos[0], j=pos[1], ny=prop.ny)
        _p = 4 * np.pi * prop.k / prop.b[ph] / prop.mu[ph]
        _p *= r_ref * pos_r[pos]
        _p /= (r_ref - pos_r[pos])

        _p *= prop.k_rel_ph_local_pressure_decision(s_1=s[pos[0], pos[1]], s_2=s[pos[0], pos[1]],
                                                    p_1=p[pos[0], pos[1]], p_2=p[pos[0], pos[1]],
                                                    ph=ph)

        j_matrix[dia_pos] = _p
    # return out.reshape((prop.nx * prop.ny, 1))


def preprocess_p(p: ResState) -> np.ndarray:
    """
    function, processes pressure
    Args:
        p: 1d array with pressures

    Returns: the same vector, but scaled

    """
    return p.v / p.bound_v / 10.0


class PetroEnv(Env):

    def _get_image(self, mode='human') -> np.ndarray:
        if mode == 'human':
            f, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
            f.tight_layout(pad=6.0)

            nx, ny = self.prop.nx, self.prop.ny
            xs = np.linspace(0, self.prop.dx * (nx - 1), nx)
            ys = np.linspace(0, self.prop.dy * (ny - 1), ny)

            label_font_size = 16
            title_font_size = 16
            x_tick_size = 14

            df = pd.DataFrame(self.p.v.reshape((nx, ny)) / 6894., columns=xs, index=ys)
            sns.heatmap(df, ax=ax[0][0], cbar=True)
            ax[0][0].set_title(f'Pressure, psi\nt={self.t: .1f} days', fontsize=title_font_size)
            ax[0][0].set_xlabel('y, m', fontsize=label_font_size)
            ax[0][0].set_ylabel('x, m', fontsize=label_font_size)
            ax[0][0].tick_params(axis='x', labelsize=x_tick_size)
            ax[0][0].tick_params(axis='y', labelsize=x_tick_size)

            df = pd.DataFrame(self.s_o.v.reshape((nx, ny)), columns=xs, index=ys)
            sns.heatmap(df, ax=ax[0][1],
                        cbar=True, fmt=".2f")
            ax[0][1].set_title(f'Saturation, oil\nt={self.t: .1f} days', fontsize=title_font_size)
            ax[0][1].set_xlabel('y, m', fontsize=label_font_size)
            ax[0][1].set_ylabel('x, m', fontsize=label_font_size)
            ax[0][1].tick_params(axis='x', labelsize=x_tick_size)
            ax[0][1].tick_params(axis='y', labelsize=x_tick_size)

            ax[1][0].bar([str(w) for w in self.pos_r], self._last_action)
            ax[1][0].set_ylim((0, 1))

            ax[1][1].bar([str(w) for w in self.pos_r], self.evaluate_wells(self._last_action))
            ax[1][1].set_ylim((0, 1))
            plt.close()

            canvas = FigureCanvas(f)

            canvas.draw()  # draw the canvas, cache the renderer
            w, h = canvas.get_width_height()
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)

            return image

    def render(self, mode='human'):
        img = self._get_image(mode=mode)
        if mode == 'human':
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.is_open

    def __init__(self, p, s_o: ResState, s_w: ResState, prop: Properties, pos_r: dict, delta_p_well: float,
                 max_time: float = 90., observation_kernel_size: int = 0, marge_4_preprocess: bool = False):
        self.max_time = max_time
        self.p_0 = p
        self.s_o_0 = s_o
        self.s_w_0 = s_w
        self.prop_0 = prop
        self.delta_p_well = delta_p_well

        self.marge_4_preprocess = marge_4_preprocess

        self.observation_kernel_size = observation_kernel_size

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
            self.delta_p_vec[two_dim_index_to_one(pos[0], pos[1], ny=prop.ny), 0] = delta_p_well
        # '''

        self.nx_ny_ones = np.ones((prop.nx * prop.ny, 1))
        self.nx_ny_eye = np.eye(prop.nx * prop.ny)
        self.t = 0
        self.laplacian_o = None
        self.dt_comp_sat = None
        self.laplacian_w = None
        self.openness = np.zeros(self.prop.nx * self.prop.ny)
        self.s_star = 0
        self.set_s_star()

        self.si_o = None
        self.si_w = None

        self.viewer = None
        self._last_action = np.ones(len(self.pos_r))

    def set_s_star(self):
        min_d = 100
        _s_star = 1
        for ss in np.linspace(0, 1, 20000):
            w_ben = self.price['w'] * self.prop.k_rel_by_ph(1 - ss, 'o') / self.prop.mu['w']
            o_ben = self.price['o'] * self.prop.k_rel_by_ph(ss, 'w') / self.prop.mu['o']
            d = abs(w_ben - o_ben)
            if d < min_d:
                _s_star = ss
                min_d = d
        self.s_star = _s_star

    def preprocess_s(self, s_o: ResState) -> np.ndarray:
        """
        normalizing saturation values. centered to values - zero benefit saturation. scaling - by 0.3
        Args:
            s_o: vector with saturation

        Returns: the same vector, but scaled

        """
        out = s_o.v - self.s_star
        out /= 0.5 - 0.2
        if self.marge_4_preprocess:
            out[out > 0] += 0.1
            out[out < 0] -= 0.1
        return out

    def extract_kernels(self, x: np.ndarray, pad_value: float) -> list:
        """
        Extract list of sub matrices, placed in well positions
        Args:
            x: 1d array, as ResState.values
            pad_value: value for padding

        Returns: list of square sub matrices

        """
        x = x.reshape((self.prop.nx, self.prop.ny))
        sub_matrices = []
        for w_pos in self.pos_r:
            x_sm = get_sub_matrix(x=x, k_size=self.observation_kernel_size,
                                  center=w_pos, pad_value=pad_value)
            sub_matrices.append(x_sm)
        return sub_matrices

    def get_observation(self) -> np.ndarray:
        """
        Process env state and returns it as vector
        Returns: env state as 1d array OR, if there is kernel,
                 it can be reshaped to (k_size, k_size, n_wells, 2).
                 where 2 is oil and pressure saturation

        """
        s_o_sc = self.preprocess_s(self.s_o)
        p_sc = preprocess_p(self.p)

        if self.observation_kernel_size > 0:
            sat_out = np.stack(self.extract_kernels(s_o_sc, pad_value=self.s_o.bound_v),
                               axis=2)
            pre_out = np.stack(self.extract_kernels(p_sc, pad_value=self.p.bound_v),
                               axis=2)
            out = np.stack([sat_out, pre_out], axis=3)
        else:
            sat_out = s_o_sc
            pre_out = p_sc
            out = np.stack([sat_out, pre_out], axis=1)

        return out.reshape(-1)

    def step(self, action: np.ndarray = None) -> [np.ndarray, float, bool, dict]:
        """

        Args:
            action: iterable, each value in (0, 1), length equal to number of well

        Returns: list of 4:
            next_state np.ndarray: it can be reshaped to (k_size, k_size, n_wells, 2).
                where 2 is oil and pressure saturation
            reward float: reward for particular action
            is_done bool: if it is a terminate position
            additional_info dict: some stuff for debug or insights, not learning

        """
        if action is not None:
            assert len(self.pos_r) == len(action)  # wanna same wells

        reward = self.evaluate_action(action)

        self.dt_comp_sat = self.s_o.v * self.prop.c['o'] + self.s_w.v * self.prop.c['w']
        self.dt_comp_sat += self.nx_ny_ones * self.prop.c['r']
        self.dt_comp_sat *= self.prop.dx * self.prop.dy * self.prop.d
        # do matrices for flow estimation
        get_j_matrix(p=self.p, s=self.s_o, pos_r=self.pos_r, ph='o', prop=self.prop, j_matrix=self.j_o)
        get_j_matrix(p=self.p, s=self.s_w, pos_r=self.pos_r, ph='w', prop=self.prop, j_matrix=self.j_w)
        # wells are open not full-wide
        self.openness = np.ones((self.prop.nx * self.prop.ny, 1))
        self._last_action = np.ones(len(self.pos_r))
        if action is not None:
            self._last_action = action
            for _i, well in enumerate(self.pos_r):
                self.openness[two_dim_index_to_one(well[0], well[1], self.prop.ny), 0] = action[_i]
            self.j_o *= self.openness
            self.j_w *= self.openness
        # now
        self.laplacian_w, self.si_w = get_laplace_one_ph(p=self.p, s=self.s_w, ph='w', prop=self.prop)
        self.laplacian_o, self.si_o = get_laplace_one_ph(p=self.p, s=self.s_o, ph='o', prop=self.prop)

        get_q_bound(self.p, self.s_w, 'w', self.prop, q_b=self.q_bound_w)
        get_q_bound(self.p, self.s_o, 'o', self.prop, q_b=self.q_bound_o)
        # set dt according Courant
        # self.prop.dt = 0.1 * self.prop.phi * self.dt_comp_sat.min() / (si_o + si_w)
        # matrix for implicit pressure
        a = self.prop.phi * sparse.diags(diagonals=[self.dt_comp_sat.reshape(-1)],
                                         offsets=[0])

        a = a - (self.laplacian_w + self.laplacian_o) * self.prop.dt
        # right hand state for ax = b
        b = self.prop.phi * self.dt_comp_sat * self.p.v + self.q_bound_w * self.prop.dt + self.q_bound_o * self.prop.dt
        b += (self.j_o * self.prop.b['o'] + self.j_w * self.prop.b['w']) * self.delta_p_vec * self.prop.dt
        # solve p
        p_new = sp_lin_alg.spsolve(a, b).reshape((-1, 1))
        # upd time stamp

        self.t += self.prop.dt / (60. * 60 * 24)

        a = self.nx_ny_ones + (self.prop.c['r'] + self.prop.c['o']) * (p_new - self.p.v)
        a *= self.prop.dx * self.prop.dy * self.prop.d * self.prop.phi

        b = self.prop.phi * self.prop.dx * self.prop.dy * self.prop.d * self.s_o.v
        b_add = (self.laplacian_o.dot(p_new) + self.q_bound_o + self.j_o * self.prop.b['o'] * self.delta_p_vec)
        b_add *= self.prop.dt
        b += b_add
        # upd target values
        self.s_o = ResState((b / a), self.s_o.bound_v, self.prop)
        self.s_w = ResState(self.nx_ny_ones - self.s_o.v, self.s_w.bound_v, self.prop)
        self.p = ResState(p_new, self.prop.p_0, self.prop)

        obs = self.get_observation()

        return [obs, reward, self.t > self.max_time, {}]

    def get_q(self, ph):
        out = None
        if ph == 'o':
            out = ((-1) * self.j_o * self.delta_p_vec).reshape((self.prop.nx, self.prop.ny))
        elif ph == 'w':
            out = ((-1) * self.j_w * self.delta_p_vec).reshape((self.prop.nx, self.prop.ny))
        return out * self.openness.reshape((self.prop.nx, self.prop.ny))

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
            self.delta_p_vec[two_dim_index_to_one(pos[0], pos[1], ny=self.prop.ny), 0] = self.delta_p_well
        # '''

        self.nx_ny_ones = np.ones((self.prop.nx * self.prop.ny, 1))
        self.nx_ny_eye = np.eye(self.prop.nx * self.prop.ny)
        self.t = 0
        self.laplacian_o = None
        self.dt_comp_sat = None
        self.laplacian_w = None
        self.openness = np.zeros(self.prop.nx * self.prop.ny)
        obs = self.get_observation()
        return obs

    def get_q_act(self, ph: str, action: np.ndarray) -> np.ndarray:
        j_o = np.zeros((self.prop.nx * self.prop.ny, 1))
        j_w = np.zeros((self.prop.nx * self.prop.ny, 1))
        get_j_matrix(p=self.p, s=self.s_o, pos_r=self.pos_r, ph='o', prop=self.prop, j_matrix=j_o)
        get_j_matrix(p=self.p, s=self.s_w, pos_r=self.pos_r, ph='w', prop=self.prop, j_matrix=j_w)

        openness = np.ones((self.prop.nx * self.prop.ny, 1))
        if action is not None:
            for _i, well in enumerate(self.pos_r):
                openness[two_dim_index_to_one(well[0], well[1], self.prop.ny), 0] = action[_i]
            j_o *= openness
            j_w *= openness
        out = None
        if ph == 'o':
            out = ((-1) * j_o * self.delta_p_vec).reshape((self.prop.nx, self.prop.ny))
        elif ph == 'w':
            out = ((-1) * j_w * self.delta_p_vec).reshape((self.prop.nx, self.prop.ny))
        return out * openness.reshape((self.prop.nx, self.prop.ny))

    def evaluate_wells(self, action: np.ndarray = None) -> np.ndarray:
        """
        the environment is associated with state. So this function estimates reward for given action
        Args:
            action: numpy array with openness of each well
        Returns: reward for each well as vector
        """
        if action is None:
            action = np.ones(len(self.pos_r))

        q_o = self.get_q_act('o', action)
        q_w = self.get_q_act('w', action)

        whole_out = (self.price['o'] * q_o - self.price['w'] * q_w) * self.prop.dt
        out = np.zeros(len(self.pos_r))
        for _i, w_pos in enumerate(self.pos_r):
            out[_i] = whole_out[w_pos]
        return out

    def evaluate_action(self, action: np.ndarray = None) -> float:
        """
        the environment is associated with state. So this function estimates reward for given action
        Args:
            action: numpy array with openness of each well

        Returns: reward as float
        """
        well_rewards = self.evaluate_wells(action)
        return well_rewards.sum()

    def evaluate_strategy(self, strategy='max_reward_for_each_time_step') -> float:
        out = 0
        done = False
        _ = self.reset()
        while not done:
            # decide on action
            action = self.get_action(strategy)
            _, r, done, _ = self.step(action)
            out += r
        return out

    def get_action(self, strategy):
        out = None
        if strategy == 'max_reward_for_each_time_step':
            out = self.__get_act_max_reward_for_each_time_step()
        if out is None:
            raise NotImplementedError
        return out

    def __get_act_max_reward_for_each_time_step(self):
        action = np.ones(len(self.pos_r))
        for _i, well in enumerate(self.pos_r):
            s_check = self.s_o[well]
            action[_i] = 0. if s_check < self.s_star else 1.
        return action

    def estimate_dt(self) -> float:
        """
        Function estimates max possible dt for this sates
        Returns: time as seconds
        """
        return self.prop.phi * self.dt_comp_sat.min() / (self.si_o + self.si_w)
