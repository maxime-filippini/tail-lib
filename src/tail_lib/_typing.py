import numpy as np

type NpFloat = np.float64 | np.float32 | np.float16 | np.longdouble


type ArrayAnyShape[DT: np.dtype] = np.ndarray[tuple[int, ...], DT]

type Array1D[DT: np.dtype] = np.ndarray[
    tuple[int],
    DT,
]


type ArrayAtLeast1D[DT: np.dtype] = np.ndarray[
    tuple[int, *tuple[int, ...]],
    DT,
]

type FloatArray = ArrayAnyShape[np.dtype[NpFloat]]
type FloatArrayAtLeast1D = ArrayAtLeast1D[np.dtype[NpFloat]]
type DateArray1D = Array1D[np.dtype[np.datetime64]]
