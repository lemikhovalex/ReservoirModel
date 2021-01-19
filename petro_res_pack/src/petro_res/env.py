import numpy as np


class Properties:
    def __init__(self, nx=25, ny=25, k=1e-1 * 1.987e-13, dx=3, dy=3, phi=0.4, p_0=150 * 10 ** 5, d=10, dt=1, s_0=0.4,
                 c_w=1e-6, c_o=1e-6, c_r=3e-6, mu_w=1 / 1000., mu_o=15 / 1000., b_o=1., b_w=1., l_w=2., l_o=2.,
                 s_wir=0.2, s_wor=0.8, k_rwr=0.1, k_rot=1., e_w=1., e_o=1., t_w=2., t_o=2.
                 ):
        # res propetis
        self.nx = nx
        self.ny = ny
        self.k = k
        self.dx = dx
        self.dy = dy
        self.phi = phi
        self.p_0 = p_0
        self.d = d
        self.dt = dt
        self.s_0 = {'w': 1 - s_0, 'o': s_0}
        self.c = {'w': c_w, 'o': c_o, 'r': c_r}
        self.mu = {'w': mu_w, 'o': mu_o}
        self.b = {'w': b_w, 'o': b_o}
        # relative saturation params
        self.l_w = l_w
        self.l_o = l_o
        self.s_wir = s_wir
        self.s_wor = s_wor
        self.k_rwr = k_rwr
        self.k_rot = k_rot
        self.e_w = e_w
        self.e_o = e_o
        self.t_w = t_w
        self.t_o = t_o
        self.mask_close = np.ones(nx*ny)
        for i in range(nx):
            self.mask_close[ny * i] = 0

    def get_s_wn(self, s_w):
        s_wn = (s_w - self.s_wir) / (self.s_wor - self.s_wir)
        if s_wn < 0:
            s_wn = 0
        if s_wn > 1:
            s_wn = 1
        return s_wn

    def k_rel_w(self, s_w):
        s_wn = self.get_s_wn(s_w)
        out = s_wn ** self.l_w * self.k_rwr
        out /= s_wn ** self.l_w + self.e_w * (1 - s_wn) ** self.t_w
        return out

    def k_rel_o(self, s_o):
        s_w = 1 - s_o
        s_wn = self.get_s_wn(s_w)
        out = self.k_rot * (1 - s_wn) ** self.l_o
        out /= (1 - s_wn) ** self.l_o + self.e_o * s_wn ** self.t_o
        return out

    def k_rel_ph_1val(self, s, ph):
        out = 0
        if ph == 'o':
            out = self.k_rel_o(s)
        elif ph == 'w':
            out = self.k_rel_w(s)
        return out

    def k_rel_ph(self, s_1, s_2, p_1, p_2, ph):
        """
        1st floor then ceil
        :param s_1:
        :param s_2:
        :param p_1:
        :param p_2:
        :param ph:
        :return:
        """
        out = 0
        if p_1 >= p_2:
            out = self.k_rel_ph_1val(s=s_1, ph=ph)
        elif p_1 <= p_2:
            out = self.k_rel_ph_1val(s=s_2, ph=ph)
        return out

    def get_s_wn_ph_np(self, sat_arr: np.ndarray):
        out = sat_arr - self.s_wir
        out /= (self.s_wor - self.s_wir)
        out = np.where(out >= 0, out, 0)
        out = np.where(out < 1, out, 1)
        return out

    def k_rel_w_np(self, sat_arr: np.ndarray):
        s_wn = self.get_s_wn_ph_np(sat_arr)
        out = s_wn ** self.l_w * self.k_rwr
        out /= s_wn ** self.l_w + self.e_w * (1 - s_wn) ** self.t_w
        return out

    def k_rel_o_np(self, s_o):
        s_w = 1 - s_o
        s_wn = self.get_s_wn_ph_np(s_w)
        out = self.k_rot * (1 - s_wn) ** self.l_o
        out /= (1 - s_wn) ** self.l_o + self.e_o * s_wn ** self.t_o
        return out

    def k_rel_ph_1val_np(self, s_arr, ph):
        out = None
        if ph == 'o':
            out = self.k_rel_o_np(s_arr)
        elif ph == 'w':
            out = self.k_rel_w_np(s_arr)
        return out


def two_dim_index_to_one(i: int, j: int, ny: int) -> int:
    return ny * i + j


def one_d_index_to_two(one_d: int, ny: int):
    i = int(one_d / ny)
    j = one_d % ny
    return i, j


class ResState:
    def __init__(self, values, bound_value, prop: Properties):
        self.v = values
        self.bound_v = bound_value
        self.shape = values.shape
        self.prop = prop

    def __getitem__(self, item):
        i, j = item
        diag = two_dim_index_to_one(i, j, self.prop.ny)
        if (i < 0) | (j < 0) | (i > self.prop.nx - 1) | (j > self.prop.ny - 1):
            return self.bound_v
        else:
            return self.v[diag, 0]


def get_lapl_one_ph_np(p: ResState, s, ph, prop: Properties):
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


    # TODO this is a time consuming stuff

    '''
    out_y = np.zeros((prop.nx + 1) * prop.ny)    
    for i in range(len(p_y_ext) - 1):
        if p_y_ext[i] >= p_y_ext[i + 1]:
            out_y[i] = s_y_ext[i]
        elif p_y_ext[i] <= p_y_ext[i + 1]:
            out_y[i] = s_y_ext[i + 1]
    '''
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

    lapl = scipy.sparse.diags(diagonals=[dist_dia, close_dia[1:],
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
