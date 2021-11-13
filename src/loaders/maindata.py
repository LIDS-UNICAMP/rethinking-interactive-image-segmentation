from __future__ import annotations
import os
import re
import random
from typing import Tuple, List, Set, Dict, Union, Optional

import numpy as np
import torch as th
import higra as hg
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torchvision.transforms.functional import normalize, to_tensor
from torch.nn.functional import interpolate
import multiprocessing as mp

from skimage.segmentation import relabel_sequential
from skimage import measure
from sklearn.preprocessing import LabelEncoder
from loaders import utils
from imgproc.segmentation import comp_watershed
import cv2


class Component:
    def __init__(self, ref_image, mask: np.ndarray, bbox: Tuple[int, int, int, int]):

        if mask.shape[1] != (bbox[2] - bbox[0]) or mask.shape[0] != (bbox[3] - bbox[1]):
            raise RuntimeError('`mask` shape and `bbox` does not match, %r and %r' % (mask.shape, bbox))

        self._ref_image = ref_image
        if mask.dtype != np.uint8:
            self.mask = mask.astype(np.uint8)
        else:
            self.mask = mask.copy()

        self.bbox = bbox  # (x, y, x_end, y_end)
        self.label = 'void'
        self.tensor = None  # type: th.Tensor
        self.position = np.zeros(2)
        self.embedding = None  # type: th.Tensor

    @property
    def ref_image(self) -> ReferenceImage:
        return self._ref_image

    @property
    def image(self) -> Image:
        return self._ref_image.image.crop(self.bbox)

    @property
    def gradient(self) -> np.ndarray:
        return self._ref_image.gradient[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]

    def assert_same_image(self, other: Component) -> None:
        assert self.ref_image == other.ref_image, "Components belong to different reference image."

    def __contains__(self, other: Component) -> bool:
        self.assert_same_image(other)
        return (other.bbox[0] >= self.bbox[0] and other.bbox[1] >= self.bbox[1] and
                other.bbox[2] <= self.bbox[2] and other.bbox[3] <= self.bbox[3])

    def __sub__(self, other: Component) -> Component:
        assert other in self, "Other component is not a subset of main component."
        diff_x = other.bbox[0] - self.bbox[0]
        diff_y = other.bbox[1] - self.bbox[1]
        w = other.bbox[2] - other.bbox[0]
        h = other.bbox[3] - other.bbox[1]
        new_mask = self.mask.copy()
        new_mask[diff_y:(diff_y + h), diff_x:(diff_x + w)][other.mask != 0] = 0
        return self.subset(new_mask)

    def inside_component(self, x: int, y: int) -> bool:
        if self.bbox[0] <= x < self.bbox[2] and self.bbox[1] <= y < self.bbox[3]:
            pos = (y - self.bbox[1], x - self.bbox[0])
            return self.mask[pos] != 0
        return False

    def empty(self) -> bool:
        return self.mask.sum() == 0

    def object_overlay(self) -> Image:
        ref_im = self._ref_image.image
        image = np.zeros((ref_im.height, ref_im.width, 4), dtype=np.uint8)
        image[:, :, 1] = 255  # green
        # alpha
        image[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2], 3] = 65 * self.mask
        return Image.fromarray(image)

    def subset(self, mask: Union[np.ndarray, th.Tensor]) -> Component:
        assert mask.shape == self.mask.shape
        if isinstance(mask, th.Tensor):
            mask = mask.numpy().astype(np.uint8)
        assert isinstance(mask, np.ndarray) and mask.dtype == np.uint8

        old_x, old_y, w, h = cv2.boundingRect(mask)
        # coord from original images
        fix_x, fix_y = self.bbox[0] + old_x, self.bbox[1] + old_y
        # adding margin and changing bbox format
        x, y, x_end, y_end = self.ref_image.compute_bbox(fix_x, fix_y, w, h)

        # creating new mask, messy because bbox is from original image
        new_mask = np.zeros((y_end - y, x_end - x), dtype=np.float32)
        diff_x, diff_y = fix_x - x, fix_y - y
        new_mask[diff_y:(diff_y + h), diff_x:(diff_x + w)] = mask[old_y:(old_y + h), old_x:(old_x + w)]

        return Component(self.ref_image, new_mask, (x, y, x_end, y_end))

    def same(self, other: Component) -> bool:
        return id(self) == id(other)

    def __eq__(self, other: Component) -> bool:
        if id(self) == id(other):
            return True
        if self._ref_image != other.ref_image:
            return False
        for i in range(len(self.bbox)):
            if other.bbox[i] != self.bbox[i]:
                return False
        return np.array_equal(self.mask, other.mask)

    def __ne__(self, other: Component) -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        return hash(id(self))

    def inside_bbox(self, point: Tuple[int, int]) -> bool:
        """
        @param point: (y, x)
        """
        return self.bbox[0] <= point[1] < self.bbox[2] and self.bbox[1] <= point[0] < self.bbox[3]

    def bbox_intersection(self, other: Component) -> bool:
        return (self.inside_bbox((other.bbox[1], other.bbox[0])) or
                self.inside_bbox((other.bbox[1], other.bbox[2])) or
                self.inside_bbox((other.bbox[3], other.bbox[0])) or
                self.inside_bbox((other.bbox[3], other.bbox[2])))

    def adjacent(self, other: Component) -> bool:
        if self._ref_image != other.ref_image or not self.bbox_intersection(other):
            return False
        min_x = max(self.bbox[0], other.bbox[0])
        min_y = max(self.bbox[1], other.bbox[1])
        max_x = min(self.bbox[2], other.bbox[2])
        max_y = min(self.bbox[3], other.bbox[3])

        this_bbox = (min_x - self.bbox[0], min_y - self.bbox[1],
                     max_x - self.bbox[0], max_y - self.bbox[1])
        other_bbox = (min_x - other.bbox[0], min_y - other.bbox[1],
                      max_x - other.bbox[0], max_y - other.bbox[1])

        this_subset = self.mask[this_bbox[1]:this_bbox[3], this_bbox[0]:this_bbox[2]]
        other_subset = other.mask[other_bbox[1]:other_bbox[3], other_bbox[0]:other_bbox[2]]
        assert this_subset.shape == other_subset.shape,\
            "subset array shape does not match, %r and %r found" % (this_subset.shape, other_subset.shape)
        return utils.is_adjacent(this_subset, other_subset)


class ReferenceImage:
    def __init__(self, image: Image, gradient: np.ndarray, name: str = '', directory: str = '',
                 margin: int = 10, ws_params: dict = {}):
        
        if gradient.shape[0] != image.height or gradient.shape[1] != image.width:
            raise RuntimeError('`image` and `gradient` have different shapes, %r and %r' % (gradient.shape,
                                                                                            (image.height, image.width)))

        self.image = image
        self.gradient = gradient
        self.gt = None  # type: Optional[np.ndarray]
        self.name = name
        self.dir = directory
        self.margin = margin
        self.components = []
        self.iou = 0

        self.ws_params = {
            'hierfun': hg.watershed_hierarchy_by_volume,
            'frontier': 0.1,
            'hierarchy': 'volume',
            'altitude': 1000,
        }
        self.update_params(ws_params)

    def __getitem__(self, index) -> Component:
        if not self.components:
            self.load_components()
        return self.components[index]

    def __len__(self) -> int:
        if not self.components:
            self.load_components()
        return len(self.components)

    def update_params(self, params: dict) -> None:
        self.ws_params.update(params)
        self.ws_params['hierfun'] = eval('hg.watershed_hierarchy_by_' + self.ws_params['hierarchy'])

    def compute_bbox(self, x, y, w, h) -> Tuple[int, int, int, int]:
        """
        Returns: x, y, x_end, y_end
        """
        x_end = min(self.gradient.shape[1], x + w + self.margin)
        y_end = min(self.gradient.shape[0], y + h + self.margin)
        x = max(0, x - self.margin)
        y = max(0, y - self.margin)
        return x, y, x_end, y_end

    def segmentation(self) -> np.ndarray:
        graph = hg.get_4_adjacency_graph(self.gradient.shape)
        edge_weights = hg.weight_graph(graph, self.gradient, hg.WeightFunction.mean)
        # RAG
        vertex_labels = hg.labelisation_watershed(graph, edge_weights)
        rag = hg.make_region_adjacency_graph_from_labelisation(graph, vertex_labels)
        rag_edge_weights = hg.rag_accumulate_on_edges(rag, hg.Accumulators.min, edge_weights)

        tree, altitudes = self.ws_params['hierfun'](rag, rag_edge_weights)
        hg.set_attribute(graph, 'no_border_vertex_out_degree', None)
        relevant_nodes = hg.attribute_contour_strength(tree, edge_weights) < self.ws_params['frontier']
        hg.set_attribute(graph, 'no_border_vertex_out_degree', 4)
        tree, node_map = hg.simplify_tree(tree, relevant_nodes)
        altitudes = altitudes[node_map]

        threshold = self.ws_params['altitude']
        if self.ws_params['hierarchy'] == 'dynamics':
            threshold /= 1000

        rag_vertex_labels = hg.labelisation_horizontal_cut_from_threshold(tree, altitudes, threshold)
        segments, _, _ = relabel_sequential(rag_vertex_labels[vertex_labels - 1])
        return segments

    def mask_to_comp(self, mask: np.ndarray) -> Component:
        rect = cv2.boundingRect(mask)
        x, y, x_end, y_end = self.compute_bbox(*rect)
        mask = mask[y:y_end, x:x_end].astype(np.float32)
        return Component(self, mask, (x, y, x_end, y_end))

    def load_components(self, segments: Optional[np.ndarray] = None) -> None:
        self.components = []
        if segments is not None:
            segments = measure.label(segments, background=-1, connectivity=1)
            for value in np.unique(segments):
                if value == 0:
                    continue
                mask = (segments == value).astype(np.uint8)
                comp = self.mask_to_comp(mask)
                if comp.mask.sum() > 25000:
                    new_seg = comp_watershed(comp)
                    for v in np.unique(new_seg):
                        if v == 0:
                            continue
                        new_mask = np.zeros_like(mask)
                        new_mask[comp.bbox[1]:comp.bbox[3],
                                 comp.bbox[0]:comp.bbox[2]] = (new_seg == v)
                        new_comp = self.mask_to_comp(new_mask)
                        self.components.append(new_comp)
                else:
                    self.components.append(comp)

        else:
            segments = self.segmentation()
            for value in np.unique(segments):
                mask = (segments == value).astype(np.uint8)
                self.components.append(self.mask_to_comp(mask))

    def size(self) -> Tuple[int, int]:
        return self.gradient.shape

    def get_label(self, label_map: Dict[str, int], default=None) -> np.ndarray:
        label_im = np.zeros((self.image.height, self.image.width), dtype=np.uint8)
        for comp in self.components:
            val = label_map.get(comp.label, default)
            x, y, x_end, y_end = comp.bbox
            label_im[y:y_end, x:x_end][comp.mask != 0] = val
        return label_im

    def get_color_label(self, label_map: Dict[str, Tuple[int, int, int, int]],
                        default: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        label_im = np.zeros((self.image.height, self.image.width, 4), dtype=np.uint8)
        for comp in self.components:
            val = label_map.get(comp.label, default)
            x, y, x_end, y_end = comp.bbox
            label_im[y:y_end, x:x_end][comp.mask != 0] = val
        return label_im

    def swap(self, old: Component, new: Component) -> None:
        for i, c in enumerate(self.components):
            if c == old:
                self.components[i] = new
                return
        raise Warning("Component not found.")

    def clone(self, ws_threshold: dict):
        other = ReferenceImage(self.image, self.gradient, self.name, self.dir, self.margin, ws_threshold)
        return other

    def compute_iou(self):
        if self.gt is None:
            return
        bkg = {'void': 0, 'background': 0}
        label = self.get_label(bkg, 1) 
        inter = np.logical_and(label, self.gt).sum().item()
        union = np.logical_or(label, self.gt).sum().item()
        self.iou = inter / union


class MainData(IterableDataset):
    def __init__(self, images_dir: str, auxiliary_dir: str, size: Tuple[int, int] = None,
                 pattern: str = None, sample: float = None, preload_hier: bool = True,
                 n_jobs: int = None, gts_dir: Optional[str] = None):
        super().__init__()

        self.sample = sample
        self.size = size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.images_dir = images_dir
        self.auxiliary_dir = auxiliary_dir
        self.gts_dir = gts_dir
        self.pattern = pattern

        self.bd_exts = ('.png', '.bpm', '.pgm', '.npy')
        self.im_exts = ('.jpg', '.jpeg', '.png', '.ppm')
        self.margin = 10

        self.reference_ims = []
        self.im_id = 0
        self.comp_id = 0
        self.preload = preload_hier
        self.current_limit = int(1e10)

        random.seed(42)

        self.load_parallel(n_jobs)

    def load_parallel(self, n_jobs: int = int) -> None:
        if n_jobs == 1:
            for tup in os.walk(self.images_dir):
                self.reference_ims += self.load_dir(tup)
        else:
            with mp.Pool(processes=n_jobs) as pool:
                output = pool.imap_unordered(self.load_dir, [tup for tup in os.walk(self.images_dir)])
                for ref_ims in output:
                    self.reference_ims += ref_ims

    def load_dir(self, args: Tuple) -> List[ReferenceImage]:
        dirpath, _, files = args
        if self.pattern is not None and not re.findall(self.pattern, dirpath):
            return []
            
        suffix = dirpath[len(self.images_dir):]
        if suffix and suffix[0] == '/': suffix = suffix[1:]
        aux_dirpath = os.path.join(self.auxiliary_dir, suffix)
        if self.gts_dir:
            gt_dirpath = os.path.join(self.gts_dir, suffix)
            
        files = [file for file in files if file.lower().endswith(self.im_exts)]
        if self.sample is not None:
            n_samples = int(len(files) * self.sample)
            files = random.sample(files, n_samples)

        reference_ims = []
        short_dir = dirpath.split('/')[-1]
        for file in sorted(files):
            im_path = os.path.join(dirpath, file)
            im_name = os.path.splitext(file)[0]
            boundaries_path = os.path.join(aux_dirpath, im_name)

            image = Image.open(im_path)
            try:
                weight_im = self.load_boundaries(boundaries_path)
            except Warning:
                continue

            if weight_im.shape[0] != image.height or weight_im.shape[1] != image.width:
                image = image.resize((weight_im.shape[1], weight_im.shape[0]), Image.BILINEAR)
                
            ref_image = ReferenceImage(image, weight_im, im_name, short_dir, self.margin)
            
            if self.gts_dir:
                gts_path = os.path.join(gt_dirpath, im_name)
                ref_image.gt = self.load_gts(gts_path)

            if self.preload:
                ref_image.load_components()

            reference_ims.append(ref_image)

        return reference_ims

    def load_boundaries(self, name_wo_ext: str) -> np.ndarray:
        aux_path = None
        for ext in self.bd_exts:
            aux_path = name_wo_ext + ext
            if os.path.exists(aux_path):
                break

        if not os.path.exists(aux_path):
            raise Warning('Missing %s' % aux_path)

        weight_im = np.load(aux_path) if aux_path.lower().endswith('.npy') else np.asarray(Image.open(aux_path))
        if weight_im.ndim == 3:
            weight_im = weight_im[:, :, 0]
        if weight_im.dtype != np.float32 and weight_im.dtype != np.float:
            weight_im = weight_im.astype(np.float32) / 65535
        return weight_im

    def load_gts(self, name_wo_ext: str) -> np.ndarray:
        aux_path = None
        for ext in self.bd_exts:
            aux_path = name_wo_ext + ext
            if os.path.exists(aux_path):
                break

        if not os.path.exists(aux_path):
            raise Warning('Missing %s' % aux_path)

        gt_im = np.asarray(Image.open(aux_path))
        if gt_im.ndim == 3:
            gt_im = gt_im.max(axis=2)
        else:
            gt_im = gt_im.copy()
        gt_im, _, _ = relabel_sequential(gt_im.astype(np.int))
        return gt_im

    def load_tensor(self, item: Component) -> None:
        tensor = to_tensor(item.image)
        normalize(tensor, self.mean, self.std, inplace=True)
        tensor[:, item.mask < 0.5] = 0.0
        if self.size is not None:
            tensor = tensor.view(1, *tensor.shape)
            tensor = interpolate(tensor, size=self.size, mode='bilinear', align_corners=True)
            tensor = tensor.view(tensor.shape[1:])
        item.tensor = tensor

    def __iter__(self):
        worker_info = th.utils.data.get_worker_info()
        if worker_info is not None or len(self.reference_ims) == 0:
            return None
        self.im_id = 0
        self.comp_id = 0
        return self

    def __next__(self) -> Component:
        if self.im_id >= len(self.reference_ims) or self.im_id >= self.current_limit:
            raise StopIteration
        if self.comp_id >= len(self.reference_ims[self.im_id]):
            self.im_id += 1
            self.comp_id = 0
            if self.im_id >= len(self.reference_ims) or self.im_id >= self.current_limit:
                raise StopIteration
        comp = self.reference_ims[self.im_id][self.comp_id]
        self.comp_id += 1
        if comp.tensor is None:
            self.load_tensor(comp)
        return comp

    def int_labels(self) -> np.array:
        labels_set = self.labels()
        encoder = LabelEncoder().fit(list(labels_set))
        labels_np = encoder.transform([comp.label for comp in self])
        if 'void' in labels_set:
            void_int = encoder.transform(['void'])
            labels_np[labels_np == void_int.item()] = -1
        return labels_np

    def labels(self) -> Set[str]:
        return set([comp.label for comp in self])

    def save_label_images(self, dir: str, label_map: Dict[str, int], default_label=None):
        for im in self.reference_ims:
            label_im = im.get_label(label_map, default_label)
            dir_path = os.path.join(dir, im.dir)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            pil_im = Image.fromarray(label_im)
            pil_im.save(os.path.join(dir_path, im.name + '.png'))


def comp_collate(batch):
    return batch


def get_mainloader(images_dir: str, auxiliary_dir: str, size: Tuple[int, int],
        pattern: str=r'.', sample: float=None, preload: bool=True, batch_size: int=32,
        n_jobs: int=1, gts_dir: Optional[str] = None) -> Tuple[DataLoader, MainData]:
    data = MainData(images_dir, auxiliary_dir, size, pattern, sample, n_jobs=n_jobs,
            preload_hier=preload, gts_dir=gts_dir)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=comp_collate)
    return loader, data

