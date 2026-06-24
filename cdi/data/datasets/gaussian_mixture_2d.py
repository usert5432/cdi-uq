import numpy as np
import torch

from torch.utils.data import Dataset

DEFAULT_COMPONENTS = [
    { 'center' : ( 0.000,  0.500), 'sigma' : 0.30, 'amplitude' : 1.0 },
    { 'center' : (-0.433, -0.250), 'sigma' : 0.30, 'amplitude' : 1.0 },
    { 'center' : ( 0.433, -0.250), 'sigma' : 0.30, 'amplitude' : 1.0 },
]

def _normalize_component(component):
    center = np.asarray(component['center'], dtype = np.float32)
    if center.shape != (2,):
        raise ValueError(f"Gaussian center must have shape (2,), got {center}")

    sigma = component.get('sigma', 0.2)
    sigma = np.asarray(sigma, dtype = np.float32)

    if sigma.ndim == 0:
        sigma = np.repeat(sigma, 2)

    if sigma.shape != (2,):
        raise ValueError(
            f"Gaussian sigma must be scalar or shape (2,), got {sigma}"
        )

    return {
        'center'    : center,
        'sigma'     : sigma,
        'amplitude' : float(component.get('amplitude', 1.0)),
    }

class GaussianMixture2DDataset(Dataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        split,
        n_samples,
        components       = None,
        xy_range         = (-1.0, 1.0),
        train_seed       = 0,
        test_seed        = 100_000,
        transform_image  = None,
        transform_labels = None,
    ):
        # pylint: disable=too-many-arguments
        if transform_labels is not None:
            raise ValueError(
                "GaussianMixture2DDataset does not use label transforms"
            )

        if transform_image is not None:
            raise ValueError(
                "GaussianMixture2DDataset does not use image transforms"
            )

        self._split     = split
        self._n_samples = n_samples
        self._xy_range  = tuple(float(x) for x in xy_range)

        if components is None:
            components = DEFAULT_COMPONENTS

        self._components = [
            _normalize_component(component) for component in components
        ]

        if split == 'train':
            self._base_seed = int(train_seed)
        else:
            self._base_seed = int(test_seed)

    def eval_surface_np(self, xy):
        xy = np.asarray(xy, dtype = np.float32)
        result = np.zeros(xy.shape[:-1], dtype = np.float32)

        for component in self._components:
            delta = (xy - component['center']) / component['sigma']
            dist2 = np.sum(delta * delta, axis = -1)
            result += component['amplitude'] * np.exp(-0.5 * dist2)

        return result

    def __len__(self):
        return self._n_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(self._base_seed + int(idx))

        xy  = rng.uniform(*self._xy_range, size = 2).astype(np.float32)
        h   = self.eval_surface_np(xy)

        condition = torch.tensor([ float(h) ], dtype = torch.float32)
        target    = torch.tensor(xy, dtype = torch.float32)

        return condition, target

