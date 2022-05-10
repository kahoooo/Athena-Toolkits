import numpy as np

from .functions import reduce_max, restrict_max
from .reduction_tree import ReductionTree, ReductionNode
from .typing import FloatArray

__all__ = [
    'PathIntegral',
    'Maximum',
    'Slice'
]


class AxisAligned:
    axi: int
    axj: int
    axk: int
    nxi: int
    nxj: int
    nxk: int
    xif: FloatArray
    xjf: FloatArray
    xkf: FloatArray

    ngh: int
    slci: slice
    slcj: slice
    slck: slice
    slc1: slice
    slc2: slice
    slc3: slice
    tree: ReductionTree

    nrxi: int
    nrxj: int
    nrxk: int
    ximin: float
    ximax: float
    xirat: float
    xjmin: float
    xjmax: float
    xjrat: float
    xkmin: float
    xkmax: float
    xkrat: float

    def __init__(self, data: dict, axis: int = 0) -> None:
        self.axi = 1 if axis == 0 else 0
        self.axj = 1 if axis == 2 else 2
        self.axk = axis

        self.nxi = data[f'nx{self.axi+1}']
        self.nxj = data[f'nx{self.axj+1}']
        self.nxk = data[f'nx{self.axk+1}']

        self.xif = data[f'x{self.axi+1}f']
        self.xjf = data[f'x{self.axj+1}f']
        self.xkf = data[f'x{self.axk+1}f']

        self.ngh = data['ngh']
        self.slci = slice(self.ngh, self.nxi - self.ngh)
        self.slcj = slice(self.ngh, self.nxj - self.ngh)
        self.slck = slice(self.ngh, self.nxk - self.ngh)
        self.slc1 = slice(self.ngh, data['nx1'] - self.ngh)
        self.slc2 = slice(self.ngh, data['nx2'] - self.ngh)
        self.slc3 = slice(self.ngh, data['nx3'] - self.ngh)

        self.tree = ReductionTree(
            (self.nxj - 2 * self.ngh, self.nxi - 2 * self.ngh))

        self.nrxi = data[f'nrx{self.axi+1}']
        self.nrxj = data[f'nrx{self.axj+1}']
        self.nrxk = data[f'nrx{self.axk+1}']
        self.ximin = data[f'x{self.axi+1}min']
        self.ximax = data[f'x{self.axi+1}max']
        self.xirat = data[f'x{self.axi+1}rat']
        self.xjmin = data[f'x{self.axj+1}min']
        self.xjmax = data[f'x{self.axj+1}max']
        self.xjrat = data[f'x{self.axj+1}rat']
        self.xkmin = data[f'x{self.axk+1}min']
        self.xkmax = data[f'x{self.axk+1}max']
        self.xkrat = data[f'x{self.axk+1}rat']

    def _xf_1d(self, level: int, lloc: int, nx: int, nrx: int,
               xmin: float, xmax: float, xrat: float) -> FloatArray:
        nx_block = nx - 2 * self.ngh
        nx_total = nrx << level
        x = (lloc * nx_block + np.arange(0, nx_block + 1)) / nx_total
        if xrat == 1.0:
            rw, lw = x, 1.0 - x
        else:
            ratn = xrat ** nrx
            rnx = xrat ** (x * nrx)
            lw = (rnx - ratn) / (1.0 - ratn)
            rw = 1.0 - lw
        return xmin * lw + xmax * rw

    def xf(self, node: ReductionNode) -> tuple[FloatArray, FloatArray]:
        return (self._xf_1d(node.level, node.lloc[1], self.nxi, self.nrxi,
                            self.ximin, self.ximax, self.xirat),
                self._xf_1d(node.level, node.lloc[0], self.nxj, self.nrxj,
                            self.xjmin, self.xjmax, self.xjrat))


class PathIntegral(AxisAligned):
    def __init__(self, data: dict, quantity: FloatArray,
                 axis: int = 0, min_level: int = 0) -> None:
        super().__init__(data, axis)

        nmb = data['nmb']
        levels = data['levels']
        llocs = data['llocs']

        for mb in range(nmb):
            if levels[mb] < min_level:
                continue
            node = self.tree.insert(
                levels[mb], llocs[mb, [self.axj, self.axi]])
            dx = np.expand_dims(np.diff(self.xkf[mb])[self.slck],
                                axis=(2 - self.axi, 2 - self.axj))
            partial_sum = np.sum(
                quantity[mb, self.slc3, self.slc2, self.slc1] * dx,
                axis=2 - self.axk)
            node.add(partial_sum)


class Maximum(AxisAligned):
    def __init__(self, data: dict, quantity: FloatArray,
                 axis: int = 0, min_level: int = 0) -> None:
        super().__init__(data, axis)
        self.tree.reduce_func = reduce_max
        self.tree.restrict_func = restrict_max

        nmb = data['nmb']
        levels = data['levels']
        llocs = data['llocs']

        for mb in range(nmb):
            if levels[mb] < min_level:
                continue
            node = self.tree.insert(
                levels[mb], llocs[mb, [self.axj, self.axi]])
            partial_max = np.max(
                quantity[mb, self.slc3, self.slc2, self.slc1],
                axis=2 - self.axk)
            node.add(partial_max)


class Slice(AxisAligned):
    def __init__(self, data: dict, quantity: FloatArray, x: float,
                 axis: int = 0, min_level: int = 0) -> None:
        super().__init__(data, axis)

        nmb = data['nmb']
        levels = data['levels']
        llocs = data['llocs']

        for mb in range(nmb):
            if levels[mb] < min_level:
                continue
            if (x <= self.xkf[mb, self.ngh] or
                    x > self.xkf[mb, self.nxk - self.ngh]):
                continue
            node = self.tree.insert(
                levels[mb], llocs[mb, [self.axj, self.axi]])
            k = np.searchsorted(self.xkf[mb, self.slck], x)
            sliced = np.take(quantity[mb, self.slc3, self.slc2, self.slc1],
                             k - 1, axis=2 - self.axk)
            node.add(sliced)
