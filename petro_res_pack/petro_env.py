import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_lin_alg
import pandas as pd

import warnings
from gym import Env
try:
    from gym.envs.classic_control.rendering import SimpleImageViewer
except ImportError:
    warnings.warn("Warnings for 'from gym.envs.classic_control.rendering import SimpleImageViewer'something with "
                  "linux I suppose, has no OpenGL")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns


from .properties import Properties
from .res_state import ResState
from .parsing_utils import two_dim_index_to_one
from .sub_matrices_utils import get_sub_matrix
from .math_phis_utils import get_laplace_one_ph


def get_q_bound(p: ResState, s: ResState, ph: str, prop: Properties) -> np.ndarray:
    """
    q from bounds into reservoir
    Args:
        p: pressure as ResState
        s: saturation as ResState
        ph: phase, "o" (oil) or "w" (water)
        prop: properties of reservoir

    Returns: vector dimension just as p.v.shape (nx *ny, 1)

    """
    q_b = np.zeros((prop.nx * prop.ny, 1))
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
    return q_b


def get_r_ref(prop: Properties) -> float:
    """
    For estimating flow to well it's necessary to define radius on which initial condition (solve derivatives)
    matches
    Args:
        prop: properties of reservoir

    Returns:

    """
    return 1 / (1 / prop.dx + np.pi / prop.d)


def get_j_matrix(s: ResState, p: ResState, pos_r: dict, ph: str, prop: Properties, openness: np.ndarray,
                 j_matrix: np.ndarray) -> None:
    """
    over write J matrix as from lectures
    Args:
        s: saturation of reservoir
        p: pressure in reservoir
        pos_r: dictionary position -> radius of wells
        ph: phase "o" (oil) or "w" (water)
        prop: properties of reservoir
        openness: openness of each well. 1 if in sell there is no well
        j_matrix: matrix from lectures, will be over wrote

    Returns: nothing

    """

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
    j_matrix *= openness
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
    def __init__(self, p, s_o: ResState, s_w: ResState, prop: Properties, pos_r: dict, delta_p_well: float,
                 max_time: float = 90., observation_kernel_size: int = 0, marge_4_preprocess: bool = False):
        """
        gym-like class for petro physics environment
        Args:
            p: pressure in environment, as ResSate
            s_o: saturation of oil, as ResState
            s_w:saturation of water, as ResState
            prop: properties of reservoir, as Properties
            pos_r: dictionary: position on grid of producing wells -> radius in meters
            delta_p_well: constant bottom hole pressure for well
            max_time: days, max time of env, for now only on condition of env is not terminated
            observation_kernel_size: 0, if observa full reservoir, if observe only vicinity of well, int, size of it
                                     so all returns as state are (kernell * kernell * 2 * n_wells) size
            marge_4_preprocess: if htere is marge in preprocessing, for saturation.
                                if profit rate is positive, s > 0.1 , else < -0.1
        """
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

        self.price = {'w': 5, 'o': 40}

        self.delta_p_vec = np.zeros((prop.nx * prop.ny, 1))
        for pos in pos_r:
            self.delta_p_vec[two_dim_index_to_one(pos[0], pos[1], ny=prop.ny), 0] = delta_p_well

        self.nx_ny_ones = np.ones((prop.nx * prop.ny, 1))
        self.nx_ny_eye = np.eye(prop.nx * prop.ny)
        self.t = 0
        self.openness = np.zeros((self.prop.nx * self.prop.ny, 1))
        self.s_star = 0
        self.set_s_star()

        self.estimated_dt = None
        self.viewer = None
        self._last_action = np.ones(len(self.pos_r))

    def _get_figure(self, mode: str):
        """
        constructs matplotlib figure
        Args:
            mode: mode, as for render, way of seeing the process

        Returns: just as matplotlib.subplots, figure and axes

        """
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
            return f
        else:
            raise NotImplementedError('only "human" mode(')

    def _get_image(self, mode='human') -> np.ndarray:
        """
        creates image as pixels, as array for render
        Args:
            mode: mode of viewing the env, just as regular gym.env, but now only for "human"

        Returns: array of pixels for simple viewer, as in gym.env

        """
        if mode == 'human':
            f = self._get_figure(mode=mode)

            canvas = FigureCanvas(f)

            canvas.draw()  # draw the canvas, cache the renderer
            w, h = canvas.get_width_height()
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)

            return image

    def render(self, mode='human'):
        """
        makes processes in env visible by graphics, rather hard for computing
        Args:
            mode: mode of viewing the env, just as regular gym.env, but now only for "human"

        Returns: do something with graphics, returns nothing

        """
        img = self._get_image(mode=mode)
        if mode == 'human':
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.is_open

    def set_s_star(self) -> None:
        """
        evaluates oil saturation corresponds to zero moment reward
        Returns: nothing, updates saturation

        """
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

    def __act_to_openness(self, action: np.ndarray) -> np.ndarray:
        """
        converts action (size = number of wells) to openness (size = number of cells)
        Args:
            action: well openess

        Returns:

        """
        out = np.ones((self.prop.nx * self.prop.ny, 1))
        if action is not None:
            assert len(self.pos_r) == len(action)  # wanna same wells
            for _i, well in enumerate(self.pos_r):
                out[two_dim_index_to_one(well[0], well[1], self.prop.ny), 0] = action[_i]
        return out

    def _load_action(self, action: np.ndarray = None) -> None:
        """
        gets action, save it as J-matrix for further update
        Args:
            action: well openness

        Returns: nothing, changes attributes

        """
        self.openness = self.__act_to_openness(action)
        self._last_action = self.openness
        get_j_matrix(s=self.s_o, p=self.p, pos_r=self.pos_r, ph='o', prop=self.prop, openness=self.openness,
                     j_matrix=self.j_o)
        get_j_matrix(s=self.s_w, p=self.p, pos_r=self.pos_r, ph='w', prop=self.prop, openness=self.openness,
                     j_matrix=self.j_w)

    def __get_new_pressure(self, laplacian_w, laplacian_o, dt_comp_sat, q_bound_w, q_bound_o) -> np.ndarray:
        """
        update of pressure through time-step
        Args:
            laplacian_w: laplacian-like matrix for derivatives for water saturation
            laplacian_o: laplacian-like matrix for derivatives for oil saturation
            dt_comp_sat: some constant vector from equations
            q_bound_w: water flow from bounds
            q_bound_o: oil flow from bounds

        Returns:

        """
        a = self.prop.phi * sparse.diags(diagonals=[dt_comp_sat.reshape(-1)],
                                         offsets=[0])

        a = a - (laplacian_w + laplacian_o) * self.prop.dt
        # right hand state for ax = b
        b = self.prop.phi * dt_comp_sat * self.p.v + q_bound_w * self.prop.dt + q_bound_o * self.prop.dt
        b += (self.j_o * self.prop.b['o'] + self.j_w * self.prop.b['w']) * self.delta_p_vec * self.prop.dt
        # solve p
        out = sp_lin_alg.spsolve(a, b).reshape((-1, 1))
        return out

    def __update_saturation_with_o(self, p_new, laplacian_o, q_bound_o) -> None:
        """
        updates saturation through oil
        Args:
            p_new: new pressure
            laplacian_o: laplacian-like matrix for derivatives for oil saturation
            q_bound_o: oil flow from bounds

        Returns: nothing, inner attributes are changed

        """
        a = self.nx_ny_ones + (self.prop.c['r'] + self.prop.c['o']) * (p_new - self.p.v)
        a *= self.prop.dx * self.prop.dy * self.prop.d * self.prop.phi

        b = self.prop.phi * self.prop.dx * self.prop.dy * self.prop.d * self.s_o.v
        b_add = (laplacian_o.dot(p_new) + q_bound_o + self.j_o * self.prop.b['o'] * self.delta_p_vec)
        b_add *= self.prop.dt
        b += b_add
        # upd target values
        self.s_o = ResState((b / a), self.s_o.bound_v, self.prop)
        self.s_w = ResState(self.nx_ny_ones - self.s_o.v, self.s_w.bound_v, self.prop)

    def __is_done(self) -> bool:
        """
        checks if the current state is a terminate one
        Returns: bolean. if env reached terminate state, true

        """
        return self.t > self.max_time

    def __update_state(self, action: np.ndarray) -> None:
        """
        make all magic inside one function
        Args:
            action: well openness

        Returns: nothing

        """

        self._load_action(action)

        dt_comp_sat = self.s_o.v * self.prop.c['o'] + self.s_w.v * self.prop.c['w']
        dt_comp_sat += self.nx_ny_ones * self.prop.c['r']
        dt_comp_sat *= self.prop.dx * self.prop.dy * self.prop.d
        # now
        laplacian_w, si_w = get_laplace_one_ph(p=self.p, s=self.s_w, ph='w', prop=self.prop)
        laplacian_o, si_o = get_laplace_one_ph(p=self.p, s=self.s_o, ph='o', prop=self.prop)

        self.estimated_dt = self._estimate_dt(dt_comp_sat, si_o=si_o, si_w=si_w)

        q_bound_w = get_q_bound(self.p, self.s_w, 'w', self.prop)
        q_bound_o = get_q_bound(self.p, self.s_o, 'o', self.prop)

        self.t += self.prop.dt / (60. * 60 * 24)

        # GET NEW PRESSURE
        p_new = self.__get_new_pressure(laplacian_w=laplacian_w, laplacian_o=laplacian_o, dt_comp_sat=dt_comp_sat,
                                        q_bound_o=q_bound_o, q_bound_w=q_bound_w)
        # UPDATE SATURATION
        self.__update_saturation_with_o(p_new=p_new, laplacian_o=laplacian_o, q_bound_o=q_bound_o)
        # UPDATE PRESSURE
        self.p = ResState(p_new, self.prop.p_0, self.prop)

    def step(self, action: np.ndarray = None) -> [np.ndarray, float, bool, dict]:
        """
        Most important function for RL. It takes action, makes engine stuff (time step) and returns new state, reward
        and boolean if new state is a terminate one
        Args:
            action: iterable, each value in (0, 1), length equal to number of well

        Returns: list of 4:
            next_state np.ndarray: it can be reshaped to (k_size, k_size, n_wells, 2).
                where 2 is oil and pressure saturation
            reward float: reward for particular action
            is_done bool: if it is a terminate position
            additional_info dict: some stuff for debug or insights, not learning

        """

        reward = self.evaluate_action(action)
        # UPDATE properties for calculations
        self.__update_state(action)

        obs = self.get_observation()
        done = self.__is_done()
        return [obs, reward, done, {}]

    def get_q_(self, ph: str) -> np.ndarray:
        """
        extracts flow rate for fluid from reservoir wells
        Args:
            ph: liquid, for now only water ("w") and oil ("o")

        Returns: vector of liquid rates

        """
        out = None
        if ph == 'o':
            out = ((-1) * self.j_o * self.delta_p_vec).reshape((self.prop.nx, self.prop.ny))
        elif ph == 'w':
            out = ((-1) * self.j_w * self.delta_p_vec).reshape((self.prop.nx, self.prop.ny))
        else:
            raise NotImplementedError('For now available only oil ("o") and water ("w")')
        return out

    def reset(self):
        self.p.v = np.ones((self.prop.nx * self.prop.ny, 1)) * self.p.bound_v
        self.s_o.v = np.ones((self.prop.nx * self.prop.ny, 1)) * self.prop.s_0['o']
        self.s_w.v = np.ones((self.prop.nx * self.prop.ny, 1)) * self.prop.s_0['w']
        self.pos_r = self.pos_r

        self.j_o = np.zeros((self.prop.nx * self.prop.ny, 1))
        self.j_w = np.zeros((self.prop.nx * self.prop.ny, 1))

        self.delta_p_vec = np.zeros((self.prop.nx * self.prop.ny, 1))
        for pos in self.pos_r:
            self.delta_p_vec[two_dim_index_to_one(pos[0], pos[1], ny=self.prop.ny), 0] = self.delta_p_well
        # '''

        self.nx_ny_ones = np.ones((self.prop.nx * self.prop.ny, 1))
        self.nx_ny_eye = np.eye(self.prop.nx * self.prop.ny)
        self.t = 0
        self.openness = np.zeros((self.prop.nx * self.prop.ny, 1))
        obs = self.get_observation()
        return obs

    def get_q_act(self, ph: str, action: np.ndarray) -> np.ndarray:
        """
        gets q with action
        Args:
            ph:
            action:

        Returns:

        """
        openness = self.__act_to_openness(action)
        j_o = np.zeros((self.prop.nx * self.prop.ny, 1))
        j_w = np.zeros((self.prop.nx * self.prop.ny, 1))
        get_j_matrix(s=self.s_o, p=self.p, pos_r=self.pos_r, ph='o', prop=self.prop, openness=openness, j_matrix=j_o)
        get_j_matrix(s=self.s_w, p=self.p, pos_r=self.pos_r, ph='w', prop=self.prop, openness=openness, j_matrix=j_w)

        if ph == 'o':
            out = ((-1) * j_o * self.delta_p_vec).reshape((self.prop.nx, self.prop.ny))
        elif ph == 'w':
            out = ((-1) * j_w * self.delta_p_vec).reshape((self.prop.nx, self.prop.ny))
        else:
            raise NotImplementedError('Only available for water ("w") and oil ("o")')
        return out

    def evaluate_wells(self, action: np.ndarray = None) -> np.ndarray:
        """
        the environment is associated with state. So this function estimates reward for given action
        Args:
            action: numpy array with openness of each well
        Returns: reward for each well as vector
        """
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

    def evaluate_strategy(self, strategy: str = 'max_reward_for_each_time_step') -> float:
        """
        evaluates some of build-in strategies
        Args:
            strategy: available strategies:
                "max_reward_for_each_time_step": close, if instant reward below zero

        Returns: cumulative reward

        """
        out = 0
        done = False
        _ = self.reset()
        while not done:
            action = self._get_action(strategy)
            _, r, done, _ = self.step(action)
            out += r
        return out

    def _get_action(self, strategy) -> np.ndarray:
        """
        estimates action for particular strategy
        Args:
            strategy:

        Returns:

        """
        out = None
        if strategy == 'max_reward_for_each_time_step':
            out = self.__get_act_max_reward_for_each_time_step()
        if out is None:
            raise NotImplementedError
        return out

    def __get_act_max_reward_for_each_time_step(self) -> np.ndarray:
        """
        for max instant reward
        Returns:

        """
        action = np.ones(len(self.pos_r))
        for _i, well in enumerate(self.pos_r):
            s_check = self.s_o[well]
            action[_i] = 0. if s_check < self.s_star else 1.
        return action

    def _estimate_dt(self, dt_comp_sat: np.ndarray, si_o: float, si_w: float) -> float:
        """
        Function estimates max possible dt for this sates
        Returns: time as seconds
        """
        return self.prop.phi * dt_comp_sat.min() / (si_o + si_w)
