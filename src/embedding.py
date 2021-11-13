import numpy as np
import torch as th
from torch.utils.data.dataloader import DataLoader

from networks.hrnet import HRNet
from config import Config
from networks.config import config, update_config

from umap import UMAP
from scipy.cluster.vq import kmeans2
from loaders.maindata import Component, ReferenceImage
from sklearn.preprocessing import LabelEncoder

from typing import List, Tuple, Iterator


class Embedding:
    def __init__(self, data_loader: DataLoader, network: str='hrnet', cuda: bool=False, weigths_path: str=None):
        network = network.lower()

        self.cuda = cuda
        self.data_loader = data_loader

        if network == 'hrnet':
            cfg = config
            update_config(cfg, Config.hrnet_config_path(), [])
            self.model = HRNet(cfg)
        else:
            raise NotImplementedError

        if weigths_path:
            self.model.init_weights(weigths_path)

        if self.cuda:
            self.model = self.model.cuda()

    def loadNetworkFeatures(self, recompute_all: bool = True) -> None:
        self.model.eval()
        count = 0
        with th.no_grad():
            for batch in self.data_loader:
                # skip already computed
                tensor = [comp.tensor for comp in batch if recompute_all or comp.embedding is None]
                if not tensor:
                    continue
                tensor = th.stack(tensor)
                print('\rProcessed %4d components; Current batch size %3d' % (count, len(tensor)), end='')
                count += len(tensor)
                if self.cuda:
                    tensor = tensor.cuda()
                embedding = self.model.forward(tensor).cpu().view(len(tensor), -1)
                index = 0
                for comp in batch:
                    if recompute_all or comp.embedding is None:
                        comp.embedding = embedding[index]
                        index += 1
        print('')

    def load2dProj(self) -> None:
        embedding = th.stack([comp.embedding for comp in self.data_loader.dataset])
        str_labels = np.array([comp.label for comp in self.data_loader.dataset])
        labels = LabelEncoder().fit_transform(str_labels)
        labels[str_labels == 'void'] = -1

        if embedding.ndim != 2:
            embedding = embedding.reshape(-1, embedding.shape[-1])

        n_neigh = 15
        self.umap = UMAP(n_neighbors=n_neigh, min_dist=0.01)
        projection = self.umap.fit_transform(embedding.numpy(), labels)
        for i, comp in enumerate(self.data_loader.dataset):
            comp.position[:] = projection[i]


    def load2dProjSubset(self, selected: List[Component], min_dist: float=0.1) -> np.ndarray:
        n_neigh = min(5, len(selected) - 1)
        embedding = th.stack([comp.embedding for comp in selected])
        tmp_umap = UMAP(n_neighbors=n_neigh, min_dist=min_dist)
        return tmp_umap.fit_transform(embedding.numpy())

    def kClosests(self, query_comp: Component, exclude: set = None, only: str = None,
                  k: int = 500) -> Tuple[List[float], List[Component]]:
        if exclude is None:
            exclude = {}
        embedding = []
        components = []
        for comp in self.data_loader.dataset:
            if (only is not None and comp.label == only) or comp.label not in exclude:
                embedding.append(comp.embedding)
                components.append(comp)
        embedding = th.stack(embedding)
        similarity = embedding @ query_comp.embedding
        k = min(len(similarity), k)
        values, indices = th.topk(similarity, k=k)
        return values.tolist(), [components[i] for i in indices.numpy()]

    def kMaxEntropy(self, k_scale: float, k_maximum: int = 5) -> Iterator[Component]:
        embedding = []
        components = []
        labels = set()
        centers = []
        for comp in self.data_loader.dataset:
           if comp.label != 'void':
                centers.append(comp.embedding)
                labels.add(comp.label)
           else:
                embedding.append(comp.embedding)
                components.append(comp)
 
        embedding = th.stack(embedding)
        centers = th.stack(centers)

        k_means = min(len(centers), int(k_scale * len(labels)))
        if k_means < len(embedding):
            centers, _ = kmeans2(centers.numpy(), k=k_means, minit='++', check_finite=False)
            centers = th.tensor(centers)
        else:
            centers = embedding

        similarity = embedding @ centers.T
        similarity /= similarity.sum(dim=1, keepdim=True)
        entropy = - th.sum(similarity * th.log(similarity + 1e-23), dim=1)
        k_maximum = min(len(entropy), k_maximum)
        values, indices = th.topk(entropy, k=k_maximum)
        for i in indices.numpy():
            # FIXME the component might not exist anymore
            yield components[i]

    def loadSingleComponent(self, comp: Component, embedding: bool = True, projection: bool = True) -> None:
        if embedding:
            self.model.eval()
            with th.no_grad():
                if comp.tensor is None:
                    self.data_loader.dataset.load_tensor(comp)
                tensor = comp.tensor.unsqueeze(0)
                if self.cuda:
                    tensor = tensor.cuda()
                comp.embedding = self.model.forward(tensor).cpu().squeeze_()
        if projection:
            comp.position[:] = self.umap.transform(comp.embedding.view(1, -1).numpy())[0]

    def loadImageComponents(self, image: ReferenceImage) -> None:
        batch = []
        for comp in image:
            if comp.tensor is None:
                self.data_loader.dataset.load_tensor(comp)
            batch.append(comp.tensor)
        batch = th.stack(batch)
        if self.cuda:
            batch = batch.cuda()

        self.model.eval()
        with th.no_grad():
            embedding = self.model.forward(batch).cpu().view(len(image), -1)

        projection = self.umap.transform(embedding.numpy())
        for comp, emb, proj in zip(image, embedding, projection):
            comp.embedding = emb
            comp.position = proj
