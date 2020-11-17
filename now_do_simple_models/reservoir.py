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


def get_lapl_one_ph(p: ResState, s: ResState, ph: str, prop: Properties):
    lapl = np.zeros((prop.nx * prop.ny, prop.nx * prop.ny))
    for dia in range(prop.nx * prop.ny):
        i, j = u.one_d_index_to_two(one_d=dia, ny=prop.ny)
        lapl[dia, dia] -= prop.k * prop.k_rel_ph(s_1=s[i, j - 1], s_2=s[i, j], p_1=p[i, j - 1], p_2=p[i, j], ph=ph)
        lapl[dia, dia] -= prop.k * prop.k_rel_ph(s_1=s[i, j + 1], s_2=s[i, j], p_1=p[i, j + 1], p_2=p[i, j], ph=ph)
        lapl[dia, dia] -= prop.k * prop.k_rel_ph(s_1=s[i - 1, j], s_2=s[i, j], p_1=p[i - 1, j], p_2=p[i, j], ph=ph)
        lapl[dia, dia] -= prop.k * prop.k_rel_ph(s_1=s[i + 1, j], s_2=s[i, j], p_1=p[i + 1, j], p_2=p[i, j], ph=ph)
        if j - 1 >= 0:
            lapl[dia, dia - 1] += prop.k * prop.k_rel_ph(s_1=s[i, j - 1], s_2=s[i, j],
                                                         p_1=p[i, j - 1], p_2=p[i, j],
                                                         ph=ph)
        if j + 1 < prop.ny:
            lapl[dia, dia + 1] += prop.k * prop.k_rel_ph(s_1=s[i, j + 1], s_2=s[i, j],
                                                         p_1=p[i, j + 1], p_2=p[i, j],
                                                         ph=ph)
        if i - 1 >= 0:
            dia_ = u.two_dim_index_to_one(i=i - 1, j=j, ny=prop.ny)
            lapl[dia, dia_] += prop.k * prop.k_rel_ph(s_1=s[i - 1, j], s_2=s[i, j],
                                                      p_1=p[i - 1, j], p_2=p[i, j],
                                                      ph=ph)
        if i + 1 < prop.nx:
            dia_ = u.two_dim_index_to_one(i=i + 1, j=j, ny=prop.ny)
            lapl[dia, dia_] += prop.k * prop.k_rel_ph(s_1=s[i + 1, j], s_2=s[i, j],
                                                      p_1=p[i + 1, j], p_2=p[i, j],
                                                      ph=ph)
    lapl *= prop.d * prop.dy / prop.mu[ph] / prop.dx
    return lapl


def get_q_bound(p, s, ph, prop: Properties):
    out = np.zeros((prop.nx, prop.ny))
    for row in range(prop.nx):
        # (row, -0.5)
        k_r = prop.k_rel_ph(s_1=s[row, -1], s_2=s[row, 0],
                            p_1=p[row, -1], p_2=p[row, 0],
                            ph=ph)
        out[row, 0] += prop.k * k_r / prop.dx * p[row, -0.5]
        # (row, ny-0.5)
        k_r = prop.k_rel_ph(s_1=s[row, prop.ny], s_2=s[row, prop.ny - 1],
                            p_1=p[row, prop.ny], p_2=p[row, prop.ny - 1],
                            ph=ph)
        out[row, prop.ny - 1] += prop.k * k_r / prop.dx * p[row, prop.ny - 0.5]

    for col in range(prop.ny):
        # (-0.5, col)
        k_r = prop.k_rel_ph(s_1=s[-1, col], s_2=s[0, col],
                            p_1=p[-1, col], p_2=p[0, col],
                            ph=ph)
        out[0, col] += prop.k * k_r / prop.dx * p[-0.5, col]
        # (nx-0.5, col)
        k_r = prop.k_rel_ph(s_1=s[prop.nx, col], s_2=s[prop.nx - 1, col],
                            p_1=p[prop.nx, col], p_2=p[prop.nx - 1, col],
                            ph=ph)
        out[prop.nx - 1, col] += prop.k * k_r / prop.dx * p[prop.nx - 0.5, col]
    out *= prop.d * prop.dy / prop.mu[ph]
    return out.reshape((prop.nx * prop.ny, 1))


def get_r_ref(prop: Properties):
    return 1 / (1 / prop.dx + np.pi / prop.d)


def get_j_matrix(p, s, pos_r, ph, prop: Properties):
    out = np.zeros((prop.nx, prop.ny))
    r_ref = get_r_ref(prop)
    for pos in pos_r:
        out[pos] = 4 * np.pi * prop.k / prop.b[ph] / prop.mu[ph]
        out[pos] *= r_ref * pos_r[pos]
        out[pos] /= (r_ref + pos_r[pos])
        out[pos] *= prop.k_rel_ph(s_1=s[pos[0], pos[1]], s_2=s[pos[0], pos[1]],
                                  p_1=p[pos[0], pos[1]], p_2=p[pos[0], pos[1]],
                                  ph=ph)
    return out.reshape((prop.nx * prop.ny, 1))
