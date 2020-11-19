import utils as u
from res_properties import Properties
import numpy as np
import scipy


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
    s_b_test_x = np.ones((prop.nx, 1)) * s.bound_v
    p_b_test_x = np.ones((prop.nx, 1)) * p.bound_v

    s_x_ext = np.append(s.v.reshape((prop.nx, prop.ny)), s_b_test_x, axis=1)
    s_x_ext = s_x_ext.reshape(-1)
    s_x_ext = np.insert(s_x_ext, 0, s.bound_v)

    p_x_ext = np.append(p.v.reshape((prop.nx, prop.ny)), p_b_test_x, axis=1)
    p_x_ext = p_x_ext.reshape(-1)
    p_x_ext = np.insert(p_x_ext, 0, p.bound_v)

    out_x = np.zeros(prop.nx * (prop.ny + 1))
    for i in range(len(p_x_ext) - 1):
        if p_x_ext[i] >= p_x_ext[i + 1]:
            out_x[i] = s_x_ext[i]
        else:
            out_x[i] = s_x_ext[i + 1]
    k_rel_x = np.array([prop.k_rel_ph_1val(s_o, ph) for s_o in out_x])  # consuming)
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

    out_y = np.zeros((prop.nx + 1) * prop.ny)

    for i in range(len(p_y_ext) - 1):
        if p_y_ext[i] >= p_y_ext[i + 1]:
            out_y[i] = s_y_ext[i]
        elif p_y_ext[i] <= p_y_ext[i + 1]:
            out_y[i] = s_y_ext[i + 1]

    k_rel_y = np.array([prop.k_rel_ph_1val(s_o, ph) for s_o in out_y])  # consuming
    sigma = min(sigma, k_rel_y.max())
    # k_rel_w_y = [f(1 - s_o) for s_o in out_y] # consuming

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

    lapl = scipy.sparse.diags(diagonals=[dist_dia, close_dia[1:],
                                         -1 * main_dia,
                                         close_dia[1:], dist_dia
                                         ],
                              offsets=[-1 * prop.ny, -1, 0, 1, prop.ny]).toarray()
    lapl *= prop.k * prop.d * prop.dy / prop.mu[ph] / prop.dx
    sigma *= prop.k * prop.d * prop.dy / prop.mu[ph] / prop.dx
    '''
    if np.all(lapl == 0):
        print('zero lapl')
    '''
    return lapl, sigma


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
