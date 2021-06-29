import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_lin_alg
from .properties import Properties
from .res_state import ResState
from .utils import two_dim_index_to_one


def pad_with(vector, pad_width, kwargs):
    pad_value = kwargs.get('pad_value', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def get_sub_matrix(x_, k_size, center, pad):
    x_padded_ = np.pad(x_, k_size // 2, pad_with, pad_value=pad)
    out = x_padded_[center[0]: center[0] + 2 * (k_size // 2) + 1,
                    center[1]: center[1] + 2 * (k_size // 2) + 1
                    ]
    return out


def get_laplace_one_ph_np(p: ResState, s, ph, prop: Properties):
    s_b_test_x = np.ones((prop.nx, 1)) * s.bound_v
    p_b_test_x = np.ones((prop.nx, 1)) * p.bound_v

    s_x_ext = np.append(s.v.reshape((prop.nx, prop.ny)), s_b_test_x, axis=1)
    s_x_ext = s_x_ext.reshape(-1)
    s_x_ext = np.insert(s_x_ext, 0, s.bound_v)

    p_x_ext = np.append(p.v.reshape((prop.nx, prop.ny)), p_b_test_x, axis=1)
    p_x_ext = p_x_ext.reshape(-1)
    p_x_ext = np.insert(p_x_ext, 0, p.bound_v)

    '''
    for i in range(len(p_x_ext) - 1):  # TODO this is a time consuming stuff
        if p_x_ext[i] >= p_x_ext[i + 1]:
            out_x[i] = s_x_ext[i]
        else:
            out_x[i] = s_x_ext[i + 1]
    '''
    comp_p_get_s = np.dtype({'names': ['p1', 'p2', 's1', 's2'],
                             'formats': [np.double,
                                         np.double,
                                         np.double,
                                         np.double]})
    p_df = np.zeros(len(p_x_ext) - 1, dtype=comp_p_get_s)
    p_df['p1'] = p_x_ext[:-1]
    p_df['p2'] = p_x_ext[1:]

    p_df['s1'] = s_x_ext[:-1]
    p_df['s2'] = s_x_ext[1:]

    out_x = np.where(p_df['p1'] >= p_df['p2'], p_df['s1'], p_df['s2'])

    k_rel_x = prop.k_rel_ph_1val_np(out_x, ph)
    # k_rel_x = np.array([prop.k_rel_ph_1val(s_o, ph) for s_o in out_x])  # consuming)
    sigma = k_rel_x.max()
    ##############################################
    s_b_test_y = np.ones((1, prop.ny)) * s_b_test_x[0]
    p_b_test_y = np.ones((1, prop.ny)) * p_b_test_x[0]

    s_y_ext = np.append(s.v.reshape((prop.nx, prop.ny)), s_b_test_y, axis=0)
    s_y_ext = s_y_ext.T.reshape(-1)
    s_y_ext = np.insert(s_y_ext, 0, s.bound_v)

    p_y_ext = np.append(p.v.reshape((prop.nx, prop.ny)), p_b_test_y, axis=0)
    p_y_ext = p_y_ext.T.reshape(-1)
    p_y_ext = np.insert(p_y_ext, 0, p.bound_v)

    p_df['p1'] = p_y_ext[:-1]
    p_df['p2'] = p_y_ext[1:]

    p_df['s1'] = s_y_ext[:-1]
    p_df['s2'] = s_y_ext[1:]

    out_y = np.where(p_df['p1'] >= p_df['p2'], p_df['s1'], p_df['s2'])

    k_rel_y = prop.k_rel_ph_1val_np(out_y, ph)
    # k_rel_y = np.array([prop.k_rel_ph_1val(s_o, ph) for s_o in out_y])
    sigma = min(sigma, k_rel_y.max())

    # let's go diagonals
    # main is first
    # for x we need to drop first and last col
    main_dia = np.zeros(prop.nx * prop.ny)
    main_dia += np.delete(k_rel_x.reshape((prop.nx, prop.ny + 1)), obj=-1, axis=1).reshape(-1)
    main_dia += np.delete(k_rel_x.reshape((prop.nx, prop.ny + 1)), obj=0, axis=1).reshape(-1)

    main_dia += np.delete(k_rel_y.reshape((prop.ny, prop.nx + 1)), obj=-1, axis=1).T.reshape(-1)
    main_dia += np.delete(k_rel_y.reshape((prop.ny, prop.nx + 1)), obj=0, axis=1).T.reshape(-1)

    close_dia = k_rel_x.reshape((prop.nx, prop.ny + 1))
    close_dia = np.delete(close_dia, obj=-1, axis=1)
    close_dia = close_dia.reshape(-1)
    close_dia *= prop.mask_close

    dist_dia = k_rel_y.reshape((prop.ny, prop.nx + 1))
    dist_dia = np.delete(dist_dia, obj=-1, axis=1)
    dist_dia = np.delete(dist_dia, obj=0, axis=1)
    dist_dia = dist_dia.T.reshape(-1)

    lapl = sparse.diags(diagonals=[dist_dia, close_dia[1:],
                                   -1 * main_dia,
                                   close_dia[1:], dist_dia
                                   ],
                        offsets=[-1 * prop.ny, -1, 0, 1, prop.ny])  # .toarray()
    lapl *= prop.k * prop.d * prop.dy / prop.mu[ph] / prop.dx
    sigma *= prop.k * prop.d * prop.dy / prop.mu[ph] / prop.dx
    return lapl, sigma


def get_q_bound(p: ResState, s, ph, prop: Properties, q_b):
    q_b *= 0
    for row in range(prop.nx):
        # (row, -0.5)
        k_r = prop.k_rel_ph(s_1=s[row, -1], s_2=s[row, 0],
                            p_1=p[row, -1], p_2=p[row, 0],
                            ph=ph)

        dia_ = two_dim_index_to_one(i=row, j=0, ny=prop.ny)
        q_b[dia_, 0] += prop.k * k_r / prop.dx * p[row, -0.5] * prop.d * prop.dy / prop.mu[ph]
        # (row, ny-0.5)
        k_r = prop.k_rel_ph(s_1=s[row, prop.ny - 1], s_2=s[row, prop.ny],
                            p_1=p[row, prop.ny - 1], p_2=p[row, prop.ny],
                            ph=ph)
        dia_ = two_dim_index_to_one(i=row, j=prop.ny - 1, ny=prop.ny)
        q_b[dia_, 0] += prop.k * k_r / prop.dx * p[row, prop.ny - 0.5] * prop.d * prop.dy / prop.mu[ph]

    for col in range(prop.ny):
        # (-0.5, col)
        k_r = prop.k_rel_ph(s_1=s[-1, col], s_2=s[0, col],
                            p_1=p[-1, col], p_2=p[0, col],
                            ph=ph)
        dia_ = two_dim_index_to_one(i=0, j=col, ny=prop.ny)
        q_b[dia_, 0] += prop.k * k_r / prop.dx * p[-0.5, col] * prop.d * prop.dy / prop.mu[ph]
        # (nx-0.5, col)
        k_r = prop.k_rel_ph(s_1=s[prop.nx - 1, col], s_2=s[prop.nx, col],
                            p_1=p[prop.nx - 1, col], p_2=p[prop.nx, col],
                            ph=ph)
        dia_ = two_dim_index_to_one(i=prop.nx - 1, j=col, ny=prop.ny)
        q_b[dia_, 0] += prop.k * k_r / prop.dx * p[prop.nx - 0.5, col] * prop.d * prop.dy / prop.mu[ph]
        # corners to zero
    # return q_b.reshape((prop.nx * prop.ny, 1))


def get_r_ref(prop: Properties):
    return 1 / (1 / prop.dx + np.pi / prop.d)


def get_j_matrix(s, p, pos_r, ph, prop: Properties, j_matr):
    r_ref = get_r_ref(prop)
    for pos in pos_r:
        dia_pos = two_dim_index_to_one(i=pos[0], j=pos[1], ny=prop.ny)
        _p = 4 * np.pi * prop.k / prop.b[ph] / prop.mu[ph]
        _p *= r_ref * pos_r[pos]
        _p /= (r_ref - pos_r[pos])

        _p *= prop.k_rel_ph(s_1=s[pos[0], pos[1]], s_2=s[pos[0], pos[1]],
                            p_1=p[pos[0], pos[1]], p_2=p[pos[0], pos[1]],
                            ph=ph)

        j_matr[dia_pos] = _p
    # return out.reshape((prop.nx * prop.ny, 1))


def preprocess_p(p: ResState) -> np.ndarray:
    return p.v / p.bound_v / 10.0


class PetroEnv:
    def __init__(self, p, s_o: ResState, s_w: ResState, prop: Properties, pos_r: dict, delta_p_well: float,
                 max_time: float = 90., observation_kernel_size: int = None):
        self.max_time = max_time
        self.p_0 = p
        self.s_o_0 = s_o
        self.s_w_0 = s_w
        self.prop_0 = prop
        self.delta_p_well = delta_p_well

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

        self.nxny_ones = np.ones((prop.nx * prop.ny, 1))
        self.nxny_eye = np.eye(prop.nx * prop.ny)
        self.t = 0
        self.lapl_o = None
        self.dt_comp_sat = None
        self.lapl_w = None
        self.openness = np.zeros(self.prop.nx * self.prop.ny)
        self.s_star = 0
        self.set_s_star()

        self.si_o = None
        self.si_w = None

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

    def preprocess_s(self, s_o: ResState) -> np.ndarray:
        out = s_o.v - self.s_star
        out /= 0.5 - 0.2
        return out

    def extract_kernels(self, x: np.ndarray, pad: float) -> list:
        x = x.reshape((self.prop.nx, self.prop.ny))
        sub_matrices = []
        for w_pos in self.pos_r:
            x_sm = get_sub_matrix(x_=x, k_size=self.observation_kernel_size,
                                  center=w_pos, pad=pad)
            sub_matrices.append(x_sm)
        return sub_matrices

    def get_observation(self, s_o: ResState, p: ResState) -> np.ndarray:
        s_o_sc = self.preprocess_s(s_o)
        p_sc = preprocess_p(p)

        obs_sub_matrices = []
        obs_sub_matrices.extend(self.extract_kernels(s_o_sc, pad=s_o.bound_v))
        obs_sub_matrices.extend(self.extract_kernels(p_sc, pad=p.bound_v))

        return np.concatenate(obs_sub_matrices, axis=None)

    def step(self, action: np.ndarray = None) -> list:
        """
        action is a np.ndarray with
        """
        if action is not None:
            assert len(self.pos_r) == len(action)  # wanna same wells

        reward = self.evaluate_action(action)

        self.dt_comp_sat = self.s_o.v * self.prop.c['o'] + self.s_w.v * self.prop.c['w']
        self.dt_comp_sat += self.nxny_ones * self.prop.c['r']
        self.dt_comp_sat *= self.prop.dx * self.prop.dy * self.prop.d
        # do matrixes for flow estimation
        get_j_matrix(p=self.p, s=self.s_o, pos_r=self.pos_r, ph='o', prop=self.prop, j_matr=self.j_o)
        get_j_matrix(p=self.p, s=self.s_w, pos_r=self.pos_r, ph='w', prop=self.prop, j_matr=self.j_w)
        # wells are open not full-wide
        self.openness = np.ones((self.prop.nx * self.prop.ny, 1))
        if action is not None:
            for _i, well in enumerate(self.pos_r):
                self.openness[two_dim_index_to_one(well[0], well[1], self.prop.ny), 0] = action[_i]
            self.j_o *= self.openness
            self.j_w *= self.openness
        # now
        self.lapl_w, self.si_w = get_laplace_one_ph_np(p=self.p, s=self.s_w, ph='w', prop=self.prop)
        self.lapl_o, self.si_o = get_laplace_one_ph_np(p=self.p, s=self.s_o, ph='o', prop=self.prop)

        get_q_bound(self.p, self.s_w, 'w', self.prop, q_b=self.q_bound_w)
        get_q_bound(self.p, self.s_o, 'o', self.prop, q_b=self.q_bound_o)
        # set dt accoarding Courant
        # self.prop.dt = 0.1 * self.prop.phi * self.dt_comp_sat.min() / (si_o + si_w)
        # matrix for implicit pressure
        a = self.prop.phi * sparse.diags(diagonals=[self.dt_comp_sat.reshape(-1)],
                                         offsets=[0])
        # a = self.nxny_eye *  self.prop.phi * self.dt_comp_sat
        a = a - (self.lapl_w + self.lapl_o) * self.prop.dt
        # right hand state for ax = b
        b = self.prop.phi * self.dt_comp_sat * self.p.v + self.q_bound_w * self.prop.dt + self.q_bound_o * self.prop.dt
        b += (self.j_o * self.prop.b['o'] + self.j_w * self.prop.b['w']) * self.delta_p_vec * self.prop.dt
        # solve p
        p_new = sp_lin_alg.spsolve(a, b).reshape((-1, 1))
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

        obs = self.get_observation(s_o=self.s_o, p=self.p)

        return [obs, reward, self.t > self.max_time, {}]

    def get_q(self, ph):
        out = None
        if ph == 'o':
            out = ((-1) * self.j_o * self.delta_p_vec).reshape((self.prop.nx, self.prop.ny))
        elif ph == 'w':
            out = ((-1) * self.j_w * self.delta_p_vec).reshape((self.prop.nx, self.prop.ny))
        return out * self.openness.reshape((self.prop.nx, self.prop.ny))

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
            self.delta_p_vec[two_dim_index_to_one(pos[0], pos[1], ny=self.prop.ny), 0] = self.delta_p_well
        # '''

        self.nxny_ones = np.ones((self.prop.nx * self.prop.ny, 1))
        self.nxny_eye = np.eye(self.prop.nx * self.prop.ny)
        self.t = 0
        self.lapl_o = None
        self.dt_comp_sat = None
        self.lapl_w = None
        self.openness = np.zeros(self.prop.nx * self.prop.ny)
        obs = self.get_observation(s_o=self.s_o, p=self.p)
        return obs

    def get_q_act(self, ph: str, action: np.ndarray) -> np.ndarray:
        j_o = np.zeros((self.prop.nx * self.prop.ny, 1))
        j_w = np.zeros((self.prop.nx * self.prop.ny, 1))
        get_j_matrix(p=self.p, s=self.s_o, pos_r=self.pos_r, ph='o', prop=self.prop, j_matr=j_o)
        get_j_matrix(p=self.p, s=self.s_w, pos_r=self.pos_r, ph='w', prop=self.prop, j_matr=j_w)

        openity = np.ones((self.prop.nx * self.prop.ny, 1))
        if action is not None:
            for _i, well in enumerate(self.pos_r):
                openity[two_dim_index_to_one(well[0], well[1], self.prop.ny), 0] = action[_i]
            j_o *= openity
            j_w *= openity
        out = None
        if ph == 'o':
            out = ((-1) * j_o * self.delta_p_vec).reshape((self.prop.nx, self.prop.ny))
        elif ph == 'w':
            out = ((-1) * j_w * self.delta_p_vec).reshape((self.prop.nx, self.prop.ny))
        return out * openity.reshape((self.prop.nx, self.prop.ny))

    def evaluate_action(self, action: np.ndarray = None) -> float:
        """
        the envinroment is assosiated with state. So this function estimates reward for given action
        Args:
            action: numpy array with openness of each well

        Returns: reward as float
        """

        if action is None:
            action = np.ones(len(self.pos_r))

        q_o = self.get_q_act('o', action)
        q_w = self.get_q_act('w', action)
        return (self.price['o'] * q_o.sum() - self.price['w'] * q_w.sum()) * self.prop.dt

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
