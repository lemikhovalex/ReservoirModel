import numpy as np
import matrixes as ma
import interblock_matrixes as i_ma
from properties import Constants
import utils as u


class Env:
    def __init__(self, k_2d_matrix: np.ndarray, poir_2d_matrix: np.ndarray, depth_2d_matrix: np.ndarray,
                 satur_2d_matrix: np.ndarray,
                 const: Constants, well_positions: dict,
                 two_d_well_index_rw_scale: dict,
                 boundary_cond: str = 'const_pressure'
                 ):
        # comes out of imput

        self.__nx, self.__ny = k_2d_matrix.shape
        # wells locations
        self.__two_d_well_index_rw_scale = two_d_well_index_rw_scale
        self.__wells_const_q = well_positions
        self.__wells_const_q = {u.two_dim_index_to_one(i=k[0], j=k[1], ny=self.__ny):
                                        self.__wells_const_q[k] for k in self.__wells_const_q}
        self.boundary_cond = boundary_cond
        self.__const = const
        # properties will not change at all

        self.__b_rat = self.__const.b_o() / self.__const.b_w()
        self.__dy = np.ones(self.__ny) * const.dy()
        self.__dx = np.ones(self.__nx) * const.dx()
        self.__poir_2d_matrix = poir_2d_matrix
        por_mat_diag = np.diag(poir_2d_matrix.reshape(-1))
        self.__por_inv = ma.inverse_diag(por_mat_diag)
        self.__k_2d_matrix = k_2d_matrix * const.k_avg()
        self.__depth_m = i_ma.DMatrix(depth_2d_matrix)
        self.__v_matrix = np.diag(const.dx() * const.dy() * depth_2d_matrix.reshape(-1))
        self.__v_matrix_inv = ma.inverse_diag(self.__v_matrix)
        depth_2d_matrix *= const.depth_avg()
        # some asserts for shape
        assert self.__k_2d_matrix.shape == poir_2d_matrix.shape
        assert poir_2d_matrix.shape == depth_2d_matrix.shape

        # init condition
        self.__p_vec = np.ones((self.__nx * self.__ny)).reshape((-1, 1)) * const.p_0()
        self.__s_o_vec = satur_2d_matrix.reshape((-1, 1))
        self.__s_w_vec = (np.ones(satur_2d_matrix.shape) - satur_2d_matrix).reshape((-1, 1))

        # build base matrices for computation
        # for pressure upd
        b_p_w = ma.get_b_p_w(poir_2d_matrix, const)
        self.__bpw_inv = ma.inverse_diag(b_p_w)
        # p in well is constant
        self.__p_well = self.__const.p_0() - self.__const.delta_p()
        # init q are zeros
        self.__q_o = np.zeros((self.__nx * self.__ny, 1))
        self.__q_w = np.zeros((self.__nx * self.__ny, 1))
        # some Nones
        self.__t_k_tilde = None
        self.__t_upd_k_tilde = None
        self._inv_p_upd = None
        self.__t_upd_k_s_o = None
        self.__t_upd_k_s_w = None
        self.__tri_matr_o = None
        self.__tri_matr_w = None
        self.__two_diag_dot = None
        self.__t_k_s_w = None
        self.__t_k_s_o = None
        self.__j_w = None
        self.__j_o = None
        self.__inv_p_upd = None
        self.upd_params()
        # assert self.__t_k_tilde is not None
        # assert self.__t_upd_k_tilde is not None
        # assert self._inv_p_upd is not None

    def upd_params(self):
        k_rel_vec = np.vectorize(self.__const.k_r_o)
        k_values_oil = k_rel_vec(self.__s_o_vec.reshape((self.__nx, self.__ny))) * self.__k_2d_matrix
        k_matrix_o = i_ma.KMatrix(k_values=k_values_oil, dy_matrix=self.__dy, dx_matrix=self.__dx)

        k_rel_vec = np.vectorize(self.__const.k_r_w)
        k_values_water = k_rel_vec(self.__s_w_vec.reshape((self.__nx, self.__ny))) * self.__k_2d_matrix
        k_matrix_w = i_ma.KMatrix(k_values=k_values_water, dy_matrix=self.__dy, dx_matrix=self.__dx)

        k_tilde = ma.get_k_tilde(consts=self.__const, k_oil_with_rel=k_matrix_o, k_wat_with_rel=k_matrix_w)
        self.__t_k_tilde = i_ma.TInterBlockMatrix(k_matrix=k_tilde, dx_matrix=self.__dx, dy_matrix=self.__dy,
                                                  d_matrix=self.__depth_m, boundary_condition=self.boundary_cond)
        self.__t_upd_k_tilde = ma.get_t_upd_matrix(self.__t_k_tilde)
        # for wells with const pressure
        self.__j_w = ma.get_j_matrix_w(self.__two_d_well_index_rw_scale, nx=self.__nx, ny=self.__ny,
                                       const=self.__const, k_matrix=k_matrix_w,
                                       depth=self.__depth_m, dx=self.__dx)
        self.__j_w *= self.__s_w_vec
        self.__j_o = ma.get_j_matrix_o(self.__two_d_well_index_rw_scale, nx=self.__nx, ny=self.__ny,
                                       const=self.__const, k_matrix=k_matrix_o,
                                       depth=self.__depth_m, dx=self.__dx)
        self.__j_o *= self.__s_o_vec
        # TODO what if we can speed up this inverse
        self.__inv_p_upd = np.eye(self.__nx * self.__ny, dtype=float)
        self.__inv_p_upd += self.__const.dt() * self.__bpw_inv.dot(
            self.__t_upd_k_tilde + self.__j_w + self.__j_o * self.__b_rat)
        self.__inv_p_upd = np.linalg.inv(self.__inv_p_upd)
        # saturation of oil upd
        b_s_o = ma.get_b_s_o(consts=self.__const, porosity=self.__poir_2d_matrix)
        k_s_o = ma.get_k_s_o(consts=self.__const, k_with_rel=k_matrix_o)
        self.__t_k_s_o = i_ma.TInterBlockMatrix(k_matrix=k_s_o,
                                                dx_matrix=self.__dx,
                                                dy_matrix=self.__dy,
                                                d_matrix=self.__depth_m,
                                                boundary_condition=self.boundary_cond
                                                )
        self.__t_upd_k_s_o = ma.get_t_upd_matrix(self.__t_k_s_o)
        self.__tri_matr_o = ma.diagonal_multidot([self.__por_inv, b_s_o, self.__v_matrix_inv])
        # saturation of water upd
        b_s_w = ma.get_b_s_o(consts=self.__const, porosity=self.__poir_2d_matrix)
        k_s_w = ma.get_k_s_w(consts=self.__const, k_with_rel=k_matrix_w)
        self.__t_k_s_w = i_ma.TInterBlockMatrix(k_matrix=k_s_w,
                                                dx_matrix=self.__dx,
                                                dy_matrix=self.__dy,
                                                d_matrix=self.__depth_m,
                                                boundary_condition=self.boundary_cond
                                                )
        self.__t_upd_k_s_w = ma.get_t_upd_matrix(self.__t_k_s_w)
        self.__tri_matr_w = ma.diagonal_multidot([self.__por_inv, b_s_w, self.__v_matrix_inv])
        # wor both saturation
        self.__two_diag_dot = ma.diagonal_multidot([self.__v_matrix_inv, self.__por_inv])

    def step(self):
        # q in wells
        self.upd_params()
        self.__q_w, self.__q_o = ma.get_q_well(self.__wells_const_q, s_w=self.__s_w_vec, s_o=self.__s_o_vec,
                                               nx=self.__nx, ny=self.__ny)
        self.__q_o = ma.get_q_well_total(self.__wells_const_q, nx=self.__nx, ny=self.__ny)
        self.__q_w = ma.get_q_well_total(self.__wells_const_q, nx=self.__nx, ny=self.__ny)
        # boundary conditions
        q_tilde_p = ma.get_q_bound(self.__t_k_tilde, self.__const.p_b())
        q_tilde_w = ma.get_q_bound(self.__t_k_s_w, self.__const.p_b())
        q_tilde_o = ma.get_q_bound(self.__t_k_s_o, self.__const.p_b())
        # const pressure well
        q_tilde_p_cpo = ma.get_q_bound_const_p_well(self.__j_o, self.__p_well)
        q_tilde_p_cpw = ma.get_q_bound_const_p_well(self.__j_w, self.__p_well)
        # pressure upd
        p_vec_new = self.__p_vec + self.__const.dt() * self.__bpw_inv.dot(
            self.__b_rat * self.__q_o + self.__q_w + q_tilde_p + q_tilde_p_cpo * self.__b_rat + q_tilde_p_cpw)
        p_vec_new = self.__inv_p_upd.dot(p_vec_new)
        # prepare to saturation upd
        q_o_cp = ma.get_q_const_p(j_matrix=self.__j_o, p_vec=p_vec_new, p_well=self.__p_well,
                                  two_d_well_index_rw_scale=self.__two_d_well_index_rw_scale, ny=self.__ny)
        q_w_cp = ma.get_q_const_p(j_matrix=self.__j_w, p_vec=p_vec_new, p_well=self.__p_well,
                                  two_d_well_index_rw_scale=self.__two_d_well_index_rw_scale, ny=self.__ny)
        # upd saturation
        # oil
        s_o_div = np.ones(self.__s_o_vec.shape) + (self.__const.c_o() + self.__const.c_r()) * (p_vec_new - self.__p_vec)
        s_o_div -= self.__const.dt() * self.__const.b_o() * self.__two_diag_dot.dot(self.__q_o)

        s_o_vec_new = self.__s_o_vec + self.__const.dt() * self.__const.b_o() * self.__two_diag_dot.dot(
            -1 * self.__t_upd_k_s_o.dot(p_vec_new) + q_tilde_o + q_o_cp)
        s_o_vec_new /= s_o_div
        self.__q_o += q_o_cp
        # water
        s_w_div = np.ones(self.__s_w_vec.shape) + (self.__const.c_w() + self.__const.c_r()) * (p_vec_new - self.__p_vec)
        s_w_div -= self.__const.dt() * self.__const.b_w() * self.__two_diag_dot.dot(self.__q_w)

        s_w_vec_new = self.__s_w_vec + self.__const.dt() * self.__const.b_w() * self.__two_diag_dot.dot(
            -1 * self.__t_upd_k_s_w.dot(p_vec_new) + q_tilde_w + q_w_cp)
        s_w_vec_new /= s_w_div
        self.__q_o += q_o_cp

        self.__q_w += q_w_cp
        # upd self var
        self.__p_vec = p_vec_new
        self.__s_o_vec = s_o_vec_new
        self.__s_w_vec = s_w_vec_new
        # self.__s_w_vec = np.ones(s_o_vec_new.shape) - self.__s_o_vec

    def s_w_as_2d(self):
        return self.__s_w_vec.reshape((self.__nx, self.__ny))

    def s_o_as_2d(self):
        return self.__s_o_vec.reshape((self.__nx, self.__ny))

    def p_as_2d(self):
        return self.__p_vec.reshape((self.__nx, self.__ny))

    def q_o_total(self):
        out = -1 * self.__q_o.sum()
        return out

    def q_w_total(self):
        return -1 * self.__q_w.sum()

    def set_wells_radius_ratio(self, upd_for_r_ratio):
        # firstly let's check of there is no extra wells to be opened)
        for nk in upd_for_r_ratio:
            assert nk in self.__two_d_well_index_rw_scale
        # then let's upd it
        for key in upd_for_r_ratio:
            self.__two_d_well_index_rw_scale[key] = upd_for_r_ratio[key]
        self.upd_params()

    def set_saturation_oil(self, s_o_new_matrix, upd_water=True):
        self.__s_o_vec = s_o_new_matrix.reshape((-1, 1))
        if upd_water:
            self.__s_w_vec = np.ones(self.__s_o_vec.shape) - self.__s_o_vec
