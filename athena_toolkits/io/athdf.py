from collections import defaultdict
from itertools import islice
from typing import Callable

import h5py
import numpy as np

__all__ = ['athdf']


def athdf(filename: str, quantities: list = None) -> defaultdict:
    nestedddict: Callable[[], defaultdict] = \
        lambda: defaultdict(nestedddict)
    nestedddict2dict: Callable[[defaultdict], dict] = \
        lambda x: {k: nestedddict2dict(v) if isinstance(v, defaultdict) else v
                   for k, v in x.items()}

    data = nestedddict()

    with h5py.File(filename, 'r') as f:
        data['coord'] = f.attrs['Coordinates']
        data['nmb'] = f.attrs['NumMeshBlocks']
        data['nx1'], data['nx2'], data['nx3'] = f.attrs['MeshBlockSize']
        data['nrx1'], data['nrx2'], data['nrx3'] = f.attrs['RootGridSize']
        data['x1min'], data['x1max'], data['x1rat'] = f.attrs['RootGridX1']
        data['x2min'], data['x2max'], data['x2rat'] = f.attrs['RootGridX2']
        data['x3min'], data['x3max'], data['x3rat'] = f.attrs['RootGridX3']
        data['ncycle'] = f.attrs['NumCycles']
        data['time'] = f.attrs['Time']

        data['maxlevel'] = f.attrs['MaxLevel']
        data['levels'] = f['Levels'][:]
        data['llocs'] = f['LogicalLocations'][:]

        data['x1f'] = f['x1f'][:]
        data['x2f'] = f['x2f'][:]
        data['x3f'] = f['x3f'][:]
        data['x1v'] = f['x1v'][:]
        data['x2v'] = f['x2v'][:]
        data['x3v'] = f['x3v'][:]

        data['ngh'] = np.searchsorted(data['x1f'][0], data['x1min'])

        all_var_names = [
            var_name.decode() for var_name in f.attrs['VariableNames']]
        if quantities is None:
            quantities = all_var_names
        iter_var_names = iter(all_var_names)
        for ds_name, num_var in zip(
                f.attrs['DatasetNames'], f.attrs['NumVariables']):
            ds_name = ds_name.decode()
            for i, var_name in enumerate(islice(iter_var_names, num_var)):
                if var_name in quantities:
                    data[ds_name][var_name] = f[ds_name][i]

    return nestedddict2dict(data)
