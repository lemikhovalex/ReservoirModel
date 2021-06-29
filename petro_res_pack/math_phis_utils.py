import numpy as np
from .res_state import ResState, Properties
import scipy.sparse as sparse


def one_zero_swap(x):
    return 1 - x


def get_ax_update(state: ResState, prop: Properties, axis: int) -> np.ndarray:
    if axis == 0:
        out_bound = np.ones((prop.nx, 1)) * state.bound_v
    elif axis == 1:
        out_bound = np.ones((1, prop.ny)) * state.bound_v
    else:
        assert False

    out = np.append(state.v.reshape((prop.nx, prop.ny)), out_bound, axis=one_zero_swap(axis))
    if axis == 0:
        out = out.reshape(-1)
    elif axis == 1:
        out = out.T.reshape(-1)
    out = np.insert(out, 0, state.bound_v)
    return out


def chose_sat_for_upd(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    comp_p_get_s = np.dtype({'names': ['p1', 'p2', 's1', 's2'],
                             'formats': [np.double,
                                         np.double,
                                         np.double,
                                         np.double]})
    p_df = np.zeros(len(p) - 1, dtype=comp_p_get_s)
    p_df['p1'] = p[:-1]
    p_df['p2'] = p[1:]

    p_df['s1'] = s[:-1]
    p_df['s2'] = s[1:]

    out = np.where(p_df['p1'] >= p_df['p2'], p_df['s1'], p_df['s2'])
    return out


def build_main_diagonal(k_rel_x, k_rel_y, ph: str,  prop: Properties) -> np.ndarray:
    main_dia = np.zeros(prop.nx * prop.ny)
    main_dia += np.delete(k_rel_x.reshape((prop.nx, prop.ny + 1)), obj=-1, axis=1).reshape(-1)
    main_dia += np.delete(k_rel_x.reshape((prop.nx, prop.ny + 1)), obj=0, axis=1).reshape(-1)

    main_dia += np.delete(k_rel_y.reshape((prop.ny, prop.nx + 1)), obj=-1, axis=1).T.reshape(-1)
    main_dia += np.delete(k_rel_y.reshape((prop.ny, prop.nx + 1)), obj=0, axis=1).T.reshape(-1)
    main_dia *= prop.k * prop.d * prop.dy / prop.mu[ph] / prop.dx
    return main_dia


def build_cloase_diag(k_rel_x, ph: str,  prop: Properties) -> np.ndarray:
    close_dia = k_rel_x.reshape((prop.nx, prop.ny + 1))
    close_dia = np.delete(close_dia, obj=-1, axis=1)
    close_dia = close_dia.reshape(-1)
    close_dia *= prop.mask_close
    close_dia *= prop.k * prop.d * prop.dy / prop.mu[ph] / prop.dx
    return close_dia


def build_dist_dia(k_rel_y, ph, prop):
    dist_dia = k_rel_y.reshape((prop.ny, prop.nx + 1))
    dist_dia = np.delete(dist_dia, obj=-1, axis=1)
    dist_dia = np.delete(dist_dia, obj=0, axis=1)
    dist_dia = dist_dia.T.reshape(-1)
    dist_dia *= prop.k * prop.d * prop.dy / prop.mu[ph] / prop.dx
    return dist_dia


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
    s_x_ext = get_ax_update(s, prop, axis=0)
    p_x_ext = get_ax_update(p, prop, axis=0)
    sat_x = chose_sat_for_upd(p=p_x_ext, s=s_x_ext)

    k_rel_x = prop.k_rel_ph_1val_np(sat_x, ph)
    sigma = k_rel_x.max()
    ##############################################

    s_y_ext = get_ax_update(s, prop, axis=1)
    p_y_ext = get_ax_update(p, prop, axis=1)
    sat_y = chose_sat_for_upd(p=p_y_ext, s=s_y_ext)

    k_rel_y = prop.k_rel_ph_1val_np(sat_y, ph)
    sigma = min(sigma, k_rel_y.max())

    # let's go diagonals
    # main is first
    # for x we need to drop first and last col
    main_dia = build_main_diagonal(k_rel_x, k_rel_y, ph, prop)
    close_dia = build_cloase_diag(k_rel_x, ph, prop)
    dist_dia =build_dist_dia(k_rel_y, ph, prop)

    laplacian = sparse.diags(diagonals=[dist_dia, close_dia[1:],
                                        -1 * main_dia,
                                        close_dia[1:], dist_dia
                                        ],
                             offsets=[-1 * prop.ny, -1, 0, 1, prop.ny])
    sigma *= prop.k * prop.d * prop.dy / prop.mu[ph] / prop.dx
    return laplacian, sigma
