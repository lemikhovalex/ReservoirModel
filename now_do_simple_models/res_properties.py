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
