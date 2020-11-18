import utils as u
from res_properties import Properties
import numpy as np
import multiprocessing
from mult_func import get_segm


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
        ###1#
        # 4#0#2
        ###3#
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
        k_1 = prop.k_rel_ph(s_1=s_0, s_2=s_1, p_1=p_0, p_2=p_1, ph=ph)
        k_2 = prop.k_rel_ph(s_1=s_0, s_2=s_2, p_1=p_0, p_2=p_2, ph=ph)
        k_3 = prop.k_rel_ph(s_1=s_3, s_2=s_0, p_1=p_3, p_2=p_0, ph=ph)
        k_4 = prop.k_rel_ph(s_1=s_4, s_2=s_0, p_1=p_4, p_2=p_0, ph=ph)
        lapl[dia, dia] -= prop.k * k_1
        lapl[dia, dia] -= prop.k * k_2
        lapl[dia, dia] -= prop.k * k_3
        lapl[dia, dia] -= prop.k * k_4
        if not dia_1 is None:
            lapl[dia, dia_1] += prop.k * k_1
        if not dia_2 is None:
            lapl[dia, dia_2] += prop.k * k_2
        if not dia_3 is None:
            lapl[dia, dia_3] += prop.k * k_3
        if not dia_4 is None:
            lapl[dia, dia_4] += prop.k * k_4

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


def add_peice_of_lapl(target, p: ResState, s: ResState, ph: str, prop: Properties, dia_beg, dia_end):
    # target is 1d array, just .reshape(-1)
    for dia in range(dia_beg, dia_end):
        i, j = u.one_d_index_to_two(one_d=dia, ny=prop.ny)  # indexes for 2d prop matrix
        flated_dia = u.two_dim_index_to_one(i=dia, j=dia, ny=prop.nx * prop.nx)
        target[flated_dia] -= prop.k * prop.k_rel_ph(s_1=s[i, j - 1], s_2=s[i, j], p_1=p[i, j - 1], p_2=p[i, j],
                                                     ph=ph) * \
                              prop.d * prop.dy / prop.mu[ph] / prop.dx
        target[flated_dia] -= prop.k * prop.k_rel_ph(s_1=s[i, j + 1], s_2=s[i, j], p_1=p[i, j + 1], p_2=p[i, j],
                                                     ph=ph) * \
                              prop.d * prop.dy / prop.mu[ph] / prop.dx
        target[flated_dia] -= prop.k * prop.k_rel_ph(s_1=s[i - 1, j], s_2=s[i, j], p_1=p[i - 1, j], p_2=p[i, j],
                                                     ph=ph) * \
                              prop.d * prop.dy / prop.mu[ph] / prop.dx
        target[flated_dia] -= prop.k * prop.k_rel_ph(s_1=s[i + 1, j], s_2=s[i, j], p_1=p[i + 1, j], p_2=p[i, j],
                                                     ph=ph) * \
                              prop.d * prop.dy / prop.mu[ph] / prop.dx
        if j - 1 >= 0:
            flated_dia = u.two_dim_index_to_one(i=dia, j=dia - 1, ny=prop.nx * prop.nx)
            target[flated_dia] += prop.k * prop.k_rel_ph(s_1=s[i, j - 1], s_2=s[i, j],
                                                         p_1=p[i, j - 1], p_2=p[i, j],
                                                         ph=ph) * \
                                  prop.d * prop.dy / prop.mu[ph] / prop.dx
        if j + 1 < prop.ny:
            flated_dia = u.two_dim_index_to_one(i=dia, j=dia + 1, ny=prop.nx * prop.nx)
            target[flated_dia] += prop.k * prop.k_rel_ph(s_1=s[i, j + 1], s_2=s[i, j],
                                                         p_1=p[i, j + 1], p_2=p[i, j],
                                                         ph=ph) * \
                                  prop.d * prop.dy / prop.mu[ph] / prop.dx
        if i - 1 >= 0:
            dia_ = u.two_dim_index_to_one(i=i - 1, j=j, ny=prop.ny)
            flated_dia = u.two_dim_index_to_one(i=dia, j=dia_, ny=prop.nx * prop.nx)
            target[flated_dia] += prop.k * prop.k_rel_ph(s_1=s[i - 1, j], s_2=s[i, j],
                                                         p_1=p[i - 1, j], p_2=p[i, j],
                                                         ph=ph) * \
                                  prop.d * prop.dy / prop.mu[ph] / prop.dx
        if i + 1 < prop.nx:
            dia_ = u.two_dim_index_to_one(i=i + 1, j=j, ny=prop.ny)
            flated_dia = u.two_dim_index_to_one(i=dia, j=dia_, ny=prop.nx * prop.nx)
            target[flated_dia] += prop.k * prop.k_rel_ph(s_1=s[i + 1, j], s_2=s[i, j],
                                                         p_1=p[i + 1, j], p_2=p[i, j],
                                                         ph=ph) * \
                                  prop.d * prop.dy / prop.mu[ph] / prop.dx


def get_lapl_one_ph_mth(p: ResState, s: ResState, ph: str, prop: Properties, n_th):
    # Create a list of jobs and then iterate through
    # the number of processes appending each process to
    # the job list
    jobs = []
    lapl_1d = multiprocessing.sharedctypes.Array('d', (prop.nx * prop.ny) ** 2, lock=False)
    for i, inter in enumerate(get_segm(prop.nx * prop.ny, n_th)):
        process = multiprocessing.Process(target=add_peice_of_lapl,
                                          args=(lapl_1d, p, s, ph, prop, inter[0], inter[1])
                                          )
        jobs.append(process)

        # Start the processes (i.e. calculate the random number lists)
    for j in jobs:
        j.start()

        # Ensure all of the processes have finished
    for j in jobs:
        j.join()
    return np.array(lapl_1d).reshape((prop.nx * prop.ny, prop.nx * prop.ny))
