import numpy as np
import multiprocessing

l_w = 2.
l_o = 2.
s_wir = 0.2
s_wor = 0.8
k_rwr = 0.1
k_rot = 1.
e_w = 1.
e_o = 1.
t_w = 2.
t_o = 2.
nx = 25
ny = 25


class ResState:
    def __init__(self, values, bound_value):
        self.v = values
        self.bound_v = bound_value
        self.shape = values.shape

    def __getitem__(self, item):
        i, j = item
        diag = two_dim_index_to_one(i, j, ny)
        if (i < 0) | (j < 0) | (i > nx - 1) | (j > ny - 1):
            return self.bound_v
        else:
            return self.v[diag, 0]


def two_dim_index_to_one(i: int, j: int, ny: int) -> int:
    return ny * i + j


def one_d_index_to_two(one_d: int, ny: int):
    i = int(one_d / ny)
    j = one_d % ny
    return i, j


def get_s_wn(s_w):
    s_wn = (s_w - s_wir) / (s_wor - s_wir)
    if s_wn < 0:
        s_wn = 0
    if s_wn > 1:
        s_wn = 1
    return s_wn


def k_rel_w(s_w):
    s_wn = get_s_wn(s_w)
    out = s_wn ** l_w * k_rwr
    out /= s_wn ** l_w + e_w * (1 - s_wn) ** t_w
    return out


def k_rel_o(s_o):
    s_w = 1 - s_o
    s_wn = get_s_wn(s_w)
    out = k_rot * (1 - s_wn) ** l_o
    out /= (1 - s_wn) ** l_o + e_o * s_wn ** t_o
    return out


def k_rel_ph(s_1, s_2, p_1, p_2, ph):
    out = 0
    if p_1 >= p_2:
        out = k_rel_ph_1val(s=s_1, ph=ph)
    elif p_1 <= p_2:
        out = k_rel_ph_1val(s=s_2, ph=ph)
    else:
        print('Dunnow what with k_rel_ph')
    return out


def k_rel_ph_1val(s, ph):
    out = 0
    if ph == 'o':
        out = k_rel_o(s)
    elif ph == 'w':
        out = k_rel_w(s)
    return out


def fill_zero_lapl_k_rel(target, p, s, mu, k, d, dx, dy, ph, dia_beg, dia_end):
    target[0, 0] = 1

    for dia in range(dia_beg, dia_end):
        i, j = one_d_index_to_two(one_d=dia, ny=ny)
        target[dia, dia] -= k * k_rel_ph(s_1=s[i, j-1], s_2=s[i, j], p_1=p[i, j-1], p_2=p[i, j], ph=ph)
        target[dia, dia] -= k * k_rel_ph(s_1=s[i, j+1], s_2=s[i, j], p_1=p[i, j+1], p_2=p[i, j], ph=ph)
        target[dia, dia] -= k * k_rel_ph(s_1=s[i-1, j], s_2=s[i, j], p_1=p[i-1, j], p_2=p[i, j], ph=ph)
        target[dia, dia] -= k * k_rel_ph(s_1=s[i+1, j], s_2=s[i, j], p_1=p[i+1, j], p_2=p[i, j], ph=ph)
        if j-1 >= 0:
            target[dia, dia-1] += k * k_rel_ph(s_1=s[i, j-1], s_2=s[i, j], p_1=p[i, j-1], p_2=p[i, j], ph=ph)
        if j+1 < ny:
            target[dia, dia+1] += k * k_rel_ph(s_1=s[i, j+1], s_2=s[i, j], p_1=p[i, j+1], p_2=p[i, j], ph=ph)
        if i-1 >= 0:
            dia_ = two_dim_index_to_one(i=i-1, j=j, ny=ny)
            target[dia, dia_] += k * k_rel_ph(s_1=s[i-1, j], s_2=s[i, j], p_1=p[i-1, j], p_2=p[i, j], ph=ph)
        if i+1 < nx:
            dia_ = two_dim_index_to_one(i=i+1, j=j, ny=ny)
            target[dia, dia_] += k * k_rel_ph(s_1=s[i+1, j], s_2=s[i, j], p_1=p[i+1, j], p_2=p[i, j], ph=ph)


def get_segm(n_blocks, n_th):
    step = n_blocks // n_th
    extra = n_blocks % n_th
    out = []
    current = 0
    for i in range(n_th):
        if extra > 0:
            out.append((current, current+step + 1))
            extra -= 1
            current += step + 1
        else:
            out.append((current, current+step))
            current += step
    return out


def get_lapl_one_ph_muth(p, s, mu, k, d, dx, dy, ph, n_th=8):
    """
    https://www.quantstart.com/articles/Parallelising-Python-with-Threading-and-Multiprocessing/
    """

    l_w = 2.
    l_o = 2.
    s_wir = 0.2
    s_wor = 0.8
    k_rwr = 0.1
    k_rot = 1.
    e_w = 1.
    e_o = 1.
    t_w = 2.
    t_o = 2.
    nx = 25
    ny = 25

    lapl = np.zeros((nx * ny, nx * ny))
    segm = get_segm(n_blocks=nx * ny, n_th=n_th)
    jobs = []
    for i in range(0, n_th):
        process = multiprocessing.Process(target=fill_zero_lapl_k_rel,
                                          args=(lapl, p, s, mu, k, d, dx, dy, ph, segm[i][0], segm[i][1]))
        jobs.append(process)
    for j in jobs:
        j.start()

    for j in jobs:
        j.join()
    return lapl


def worker(x):
    return x*x


def test_glob_mp(x, pos):
    x[pos] = 1



