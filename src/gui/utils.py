from contextlib import contextmanager

from PySide2.QtCore import Qt
from PySide2.QtWidgets import QApplication
from PySide2.QtGui import QImage, QCursor

import numpy as np
import qimage2ndarray as q2np
from loaders.maindata import Component
from typing import Union

from collections import OrderedDict, Callable


def q_fill_alpha(image: QImage, alpha: QImage) -> None:
    """
    Fills an image alpha band with values from the
    brightness values of the reference image.

    Parameters
    ----------
    image : RGBA QImage
    alpha : Grayscale QImage
    """
    if image.size() != alpha.size():
        raise ValueError('Alpha and Image sizes are not equal.')
    brightness = q2np.byte_view(alpha)[:, :, 0]
    alpha_view = q2np.alpha_view(image)
    alpha_view.flat[:] = brightness.flat[:]


def np_to_argb_qimage(image: np.ndarray,
                      alpha: Union[float, np.ndarray] = 1.0) -> QImage:
    data = np.empty((*image.shape[:2], 4))
    data[:, :, :3] = image
    data[:, :, 3] = 255 * alpha
    return q2np.array2qimage(data)


def mask_to_red(mask: np.ndarray) -> QImage:
    red = np.zeros((*mask.shape, 4), dtype=np.uint8)
    red[mask, ...] = (255, 0, 0, 255)
    return q2np.array2qimage(red)


def component_to_qimage(item: Component) -> QImage:
    data = np.empty((*item.mask.shape, 4))
    data[:, :, :3] = np.array(item.image)
    data[:, :, 3] = 255 * item.mask
    return q2np.array2qimage(data)


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))


@contextmanager
def wait_cursor():
    try:
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        yield
    finally:
        QApplication.restoreOverrideCursor()

