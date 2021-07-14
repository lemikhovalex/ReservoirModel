import numpy as np
from typing import Union


class Properties:

    def __init__(self, nx: int = 25, ny: int = 25, k: float = 1e-1 * 1.987e-13, dx: float = 3, dy: float = 3,
                 phi: float = 0.4, p_0: float = 150 * 10 ** 5, d: float = 10, dt: float = 24316,
                 s_0: float = 0.4, c_w: float = 1e-6, c_o: float = 1e-6, c_r: float = 3e-6, mu_w: float = 1 / 1000.,
                 mu_o: float = 15 / 1000., b_o: float = 1., b_w: float = 1., l_w: float = 2., l_o: float = 2.,
                 s_wir: float = 0.2, s_wor: float = 0.8, k_rwr: float = 0.1, k_rot: float = 1., e_w: float = 1.,
                 e_o: float = 1., t_w: float = 2., t_o: float = 2.
                 ):
        """
        This class stores reservoir properties and calculates relative permeability
        All values in metric system! No psi, darcy ect
        Args:
            nx: number of x-axis cells
            ny: number of y-axis cells
            k: absolute permeability
            dx: size of cell (x and y axis)
            dy: must be equal to dx
            phi: porosity of reservoir
            p_0: initial pressure, same for all cells
            d: depth in meters
            dt: time step, seconds
            s_0: initial oil saturation
            c_w: compressibility of water
            c_o: compressibility of oil
            c_r: compressibility of rock
            mu_w: viscosity of water
            mu_o: viscosity of oil
            b_o: oil volume below / oil volume above
            b_w: water volume below / water volume above
            l_w: parameter from https://en.wikipedia.org/wiki/Relative_permeability#LET-model
            l_o: parameter from https://en.wikipedia.org/wiki/Relative_permeability#LET-model
            s_wir: parameter from https://en.wikipedia.org/wiki/Relative_permeability#LET-model
            s_wor: parameter from https://en.wikipedia.org/wiki/Relative_permeability#LET-model
            k_rwr: parameter from https://en.wikipedia.org/wiki/Relative_permeability#LET-model
            k_rot: parameter from https://en.wikipedia.org/wiki/Relative_permeability#LET-model
            e_w: parameter from https://en.wikipedia.org/wiki/Relative_permeability#LET-model
            e_o: parameter from https://en.wikipedia.org/wiki/Relative_permeability#LET-model
            t_w: parameter from https://en.wikipedia.org/wiki/Relative_permeability#LET-model
            t_o: parameter from https://en.wikipedia.org/wiki/Relative_permeability#LET-model
        """
        self.nx = nx
        self.ny = ny
        self.k = k
        if dx != dy:
            raise ResourceWarning('dx and dy params must be equal! set both to dx')
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
        self.mask_close = np.ones(nx * ny)
        for i in range(nx):
            self.mask_close[ny * i] = 0

    def __get_s_wn(self, s_w: Union[float, np.ndarray]):
        """
        calculates parameter for relative permeability
        https://en.wikipedia.org/wiki/Relative_permeability#LET-model
        Args:
            s_w: water saturation

        Returns:

        """
        s_wn = (s_w - self.s_wir) / (self.s_wor - self.s_wir)
        if isinstance(s_wn, float) or isinstance(s_wn, int):
            if s_wn < 0:
                s_wn = 0
            if s_wn > 1:
                s_wn = 1
        elif isinstance(s_wn, np.ndarray):
            s_wn[s_wn < 0] = 0
            s_wn[s_wn > 1] = 1
        return s_wn

    def __k_rel_w(self, s_w: Union[float, np.ndarray]):
        """
        relative water permeability by single value or np.ndarray
        Args:
            s_w: water saturation

        Returns:

        """
        s_wn = self.__get_s_wn(s_w)
        out = s_wn ** self.l_w * self.k_rwr
        out /= s_wn ** self.l_w + self.e_w * (1 - s_wn) ** self.t_w
        return out

    def __k_rel_o(self, s_o: Union[float, np.ndarray]):
        """
        relative oil permeability by single value or np.ndarray
        Args:
            s_o:

        Returns:

        """
        s_w = 1 - s_o
        s_wn = self.__get_s_wn(s_w)
        out = self.k_rot * (1 - s_wn) ** self.l_o
        out /= (1 - s_wn) ** self.l_o + self.e_o * s_wn ** self.t_o
        return out

    def k_rel_by_ph(self, s: Union[np.ndarray, float], ph: str):
        """
        calculates relative permeability for given vector of saturation for given phase/liquid
        Args:
            s: saturation vector
            ph: phase, oil or water

        Returns: vector of relative permeabilities, same size as input

        """
        out = 0
        if ph == 'o':
            out = self.__k_rel_o(s)
        elif ph == 'w':
            out = self.__k_rel_w(s)
        return out

    def k_rel_ph_local_pressure_decision(self, s_1: float, s_2: float, p_1: float, p_2: float, ph: str) -> float:
        """
        there are 2 neighbouring cells, and relative permeability depends on direction of flow
        or the pressure value.
        :param s_1: saturation in 1st cell
        :param s_2: saturation in 2nd cell
        :param p_1: pressure in 1st cell
        :param p_2: pressure in 2nd cell
        :param ph: phase oil or water
        :return: float value of relative permeability
        """
        out = 0
        if p_1 >= p_2:
            out = self.k_rel_by_ph(s=s_1, ph=ph)
        elif p_1 <= p_2:
            out = self.k_rel_by_ph(s=s_2, ph=ph)
        return out
