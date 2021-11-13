import cv2
import numpy as np
import numba as nb
from PIL import Image
from typing import Tuple

from loaders.maindata import ReferenceImage, Component


def boolean_to_pil(image: np.ndarray, color: Tuple[int, int, int, int]) -> Image:
    assert image.dtype == np.bool
    assert image.ndim == 2
    colored = np.zeros((*image.shape, 4), dtype=np.uint8)
    color = np.array(color)
    colored[image, ...] = color
    return Image.fromarray(colored)


def mask_to_pil_contour(mask: np.ndarray, color: Tuple[int, int, int, int], thickness: int = 1) -> Image:
    _, contours, _ = cv2.findContours(mask, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    contour_im = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_im, contours, -1, (1,), thickness)
    return boolean_to_pil(contour_im.astype(np.bool), color)


def retrive_image_contour(image: ReferenceImage,
                          color: Tuple[int, int, int, int] = (255, 0, 0, 255),
                          thickness: int = 1) -> Image:
    segments = np.zeros(image.gradient.shape, dtype=np.int32)
    for i, c in enumerate(image):
        x, y, x_end, y_end = c.bbox
        segments[y:y_end, x:x_end] += (i + 1) * c.mask
    return mask_to_pil_contour(segments, color, thickness)


def retrive_comp_contour(comp: Component, color: Tuple[int, int, int, int] = (0, 255, 0, 255)) -> Image:
    segments = np.zeros(comp.ref_image.gradient.shape, dtype=np.int32)
    segments[comp.bbox[1]:comp.bbox[3], comp.bbox[0]:comp.bbox[2]] = comp.mask
    return mask_to_pil_contour(segments, color)
