import numpy as np
from .res_state import ResState, Properties
import scipy.sparse as sparse


def get_laplace_one_ph(p: ResState, s: ResState, ph: str, prop: Properties) -> [np.ndarray, float]:
    """
    Function creates laplacian matrix for given reservoir state
    Args:
        p: reservoir pressure
        s: reservoir saturation
        ph: phase, 'o' or 'w'
        prop: properties of reservoir

    Returns: laplacian matrix for update and sigma - important value for time step estimation

    """
    s_b_test_x = np.ones((prop.nx, 1)) * s.bound_v
    p_b_test_x = np.ones((prop.nx, 1)) * p.bound_v

    s_x_ext = np.append(s.v.reshape((prop.nx, prop.ny)), s_b_test_x, axis=1)
    s_x_ext = s_x_ext.reshape(-1)
    s_x_ext = np.insert(s_x_ext, 0, s.bound_v)

    p_x_ext = np.append(p.v.reshape((prop.nx, prop.ny)), p_b_test_x, axis=1)
    p_x_ext = p_x_ext.reshape(-1)
    p_x_ext = np.insert(p_x_ext, 0, p.bound_v)

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
                        offsets=[-1 * prop.ny, -1, 0, 1, prop.ny])
    lapl *= prop.k * prop.d * prop.dy / prop.mu[ph] / prop.dx
    sigma *= prop.k * prop.d * prop.dy / prop.mu[ph] / prop.dx
    return lapl, sigma