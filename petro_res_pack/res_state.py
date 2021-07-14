import numpy as np
from .properties import Properties
from .parsing_utils import two_dim_index_to_one


class ResState:
    def __init__(self, values: np.ndarray, bound_value: float, prop: Properties = Properties()):
        """
        Storage for reservoir values. It can store value both as one big vector (fast way and access by .v[i, 0])
        or access elements as 2d matrix (access by [i, j])
        Args:
            values: 2d array of shape (d, 1) or 1d
            bound_value: single bound values for reservoir value (pressure or saturation)
            prop: properties. there dimension is stored
        """
        self.v = values.reshape(-1, 1)
        self.bound_v = bound_value
        self.prop = prop
        if self.v.shape[0] != self.prop.nx * self.prop.ny:
            raise IndexError("wrong size of input property")

    def __getitem__(self, item) -> float:
        i, j = item
        if (i >= 0) & (j >= 0) & (i <= self.prop.nx - 1) & (j <= self.prop.ny - 1):
            # inner box where values are standard
            one_d_index = two_dim_index_to_one(i, j, self.prop.ny)
            out = self.v[one_d_index, 0]
        elif (i >= -0.5) & (j >= -0.5) & (i <= self.prop.nx + 0.5) & (j <= self.prop.ny + 0.5):
            # little out boarder with tiny margin of size 1 for values
            out = self.bound_v
        else:
            raise IndexError("It is too far from bound, extra space for board value is 0.5, not grater. 0.5 is ok")

        return out
