import utils as u
from res_properties import Properties
import numpy as np


class ResState:
    def __init__(self, values, bound_value, prop: Properties):
        self.v = values
        self.bound_v = bound_value
        self.shape = values.shape
        self.prop = prop

    def __getitem__(self, item):
        i, j = item
        diag = u.two_dim_index_to_one(i, j, self.prop.ny)
        if (i < 0) | (j < 0) | (i > self.prop.nx - 1) | (j > self.prop.ny - 1):
            return self.bound_v
        else:
            return self.v[diag, 0]


def get_lapl_one_ph(p: ResState, s: ResState, ph: str, prop: Properties, lapl):
    for dia in range(prop.nx * prop.ny):
        # ###1#
        # #4#0#2
        # ###3#
        dia_1, dia_2, dia_3, dia_4 = [None] * 4
        p_0, p_1, p_2, p_3, p_4 = [p.bound_v] * 5
        s_0, s_1, s_2, s_3, s_4 = [s.bound_v] * 5
        k_1, k_2, k_3, k_4 = [0] * 4
        i, j = u.one_d_index_to_two(one_d=dia, ny=prop.ny)
        if i - 1 >= 0:
            dia_1 = u.two_dim_index_to_one(i=i - 1, j=j, ny=prop.ny)
            p_1 = p.v[dia_1, 0]
            s_1 = s.v[dia_1, 0]
        if j + 1 < prop.ny:
            dia_2 = u.two_dim_index_to_one(i=i, j=j + 1, ny=prop.ny)
            p_2 = p.v[dia_2, 0]
            s_2 = s.v[dia_2, 0]
        if i + 1 < prop.nx:
            dia_3 = u.two_dim_index_to_one(i=i + 1, j=j, ny=prop.ny)
            p_3 = p.v[dia_3, 0]
            s_3 = s.v[dia_3, 0]
        if j - 1 >= 0:
            dia_4 = u.two_dim_index_to_one(i=i, j=j - 1, ny=prop.ny)
            p_4 = p.v[dia_4, 0]
            s_4 = s.v[dia_4, 0]

        s_0 = s.v[dia, 0]
        p_0 = p.v[dia, 0]
        k_1 = prop.k_rel_ph(s_1=s_1, s_2=s_0, p_1=p_1, p_2=p_0, ph=ph)
        k_2 = prop.k_rel_ph(s_1=s_0, s_2=s_2, p_1=p_0, p_2=p_2, ph=ph)
        k_3 = prop.k_rel_ph(s_1=s_0, s_2=s_3, p_1=p_0, p_2=p_3, ph=ph)
        k_4 = prop.k_rel_ph(s_1=s_4, s_2=s_0, p_1=p_4, p_2=p_0, ph=ph)
        lapl[dia, dia] = -1 * prop.k * k_1
        lapl[dia, dia] -= prop.k * k_2
        lapl[dia, dia] -= prop.k * k_3
        lapl[dia, dia] -= prop.k * k_4
        if dia_1 is not None:
            lapl[dia, dia_1] = prop.k * k_1
        if dia_2 is not None:
            lapl[dia, dia_2] = prop.k * k_2
        if dia_3 is not None:
            lapl[dia, dia_3] = prop.k * k_3
        if dia_4 is not None:
            lapl[dia, dia_4] = prop.k * k_4

    lapl *= prop.d * prop.dy / prop.mu[ph] / prop.dx
    # return lapl


def get_q_bound(p: ResState, s, ph, prop: Properties, q_b):
    q_b *= 0
    for row in range(prop.nx):
        # (row, -0.5)
        k_r = prop.k_rel_ph(s_1=s[row, -1], s_2=s[row, 0],
                            p_1=p[row, -1], p_2=p[row, 0],
                            ph=ph)

        dia_ = u.two_dim_index_to_one(i=row, j=0, ny=prop.ny)
        q_b[dia_, 0] += prop.k * k_r / prop.dx * p[row, -0.5] * prop.d * prop.dy / prop.mu[ph]
        # (row, ny-0.5)
        k_r = prop.k_rel_ph(s_1=s[row, prop.ny - 1], s_2=s[row, prop.ny],
                            p_1=p[row, prop.ny - 1], p_2=p[row, prop.ny],
                            ph=ph)
        dia_ = u.two_dim_index_to_one(i=row, j=prop.ny - 1, ny=prop.ny)
        q_b[dia_, 0] += prop.k * k_r / prop.dx * p[row, prop.ny - 0.5] * prop.d * prop.dy / prop.mu[ph]

    for col in range(prop.ny):
        # (-0.5, col)
        k_r = prop.k_rel_ph(s_1=s[-1, col], s_2=s[0, col],
                            p_1=p[-1, col], p_2=p[0, col],
                            ph=ph)
        dia_ = u.two_dim_index_to_one(i=0, j=col, ny=prop.ny)
        q_b[dia_, 0] += prop.k * k_r / prop.dx * p[-0.5, col] * prop.d * prop.dy / prop.mu[ph]
        # (nx-0.5, col)
        k_r = prop.k_rel_ph(s_1=s[prop.nx - 1, col], s_2=s[prop.nx, col],
                            p_1=p[prop.nx - 1, col], p_2=p[prop.nx, col],
                            ph=ph)
        dia_ = u.two_dim_index_to_one(i=prop.nx - 1, j=col, ny=prop.ny)
        q_b[dia_, 0] += prop.k * k_r / prop.dx * p[prop.nx - 0.5, col] * prop.d * prop.dy / prop.mu[ph]
        # corners to zero
    # return q_b.reshape((prop.nx * prop.ny, 1))


def get_r_ref(prop: Properties):
    return 1 / (1 / prop.dx + np.pi / prop.d)


def get_j_matrix(p, s, pos_r, ph, prop: Properties, j_matr):
    r_ref = get_r_ref(prop)
    for pos in pos_r:
        dia_pos = u.two_dim_index_to_one(i=pos[0], j=pos[1], ny=prop.ny)
        j_matr[dia_pos] = 4 * np.pi * prop.k / prop.b[ph] / prop.mu[ph]
        j_matr[dia_pos] *= r_ref * pos_r[pos]
        j_matr[dia_pos] /= (r_ref + pos_r[pos])
        j_matr[dia_pos] *= prop.k_rel_ph(s_1=s[pos[0], pos[1]], s_2=s[pos[0], pos[1]],
                                         p_1=p[pos[0], pos[1]], p_2=p[pos[0], pos[1]],
                                         ph=ph)
    # return out.reshape((prop.nx * prop.ny, 1))
