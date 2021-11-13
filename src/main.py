#!/usr/bin/env python3

import sys
import os
from datetime import datetime

os.environ['QT_MAC_WANTS_LAYER'] = '1'  # required by mac M1

import importlib
from typing import Tuple, List, Iterator
import pickle
from config import Config
import time

from PySide2.QtCore import Qt, Slot, QDir, QSize
from PySide2.QtGui import (QStandardItemModel, QStandardItem, QPixmap, QIcon,
                           QBrush, QKeyEvent, )
from PySide2.QtWidgets import (QApplication, QMainWindow, QAction, QWidget,
                               QFileDialog, QMenuBar, QListView,
                               QHBoxLayout, )

from PIL.ImageQt import ImageQt
import qimage2ndarray as q2np

from gui.projectionframe import ProjectionFrame
from gui.projsubsetframe import ProjSubsetFrame
from gui.commandsDialog import CommandsDialog
from gui.imageframe import ImageFrame
from gui.utils import component_to_qimage, wait_cursor, mask_to_red

from loaders.maindata import get_mainloader, Component, ReferenceImage
from loaders.balanceddata import get_balancedloader
from loaders.labelcityscape import labels as CS_LABELS

import optimizers.metriclearning as ml

import imgproc.contour as cont
import imgproc.segmentation as im_seg

from embedding import Embedding


class MainWindow(QMainWindow):
    def __init__(self, args=None, parent=None):
        super().__init__(parent)
        
        self.setupUI()
        self.setupActions()
        self.setupSignals()

        self.im_dir = ...  # type: QDir
        self.bd_dir = ...  # type: QDir

        self.im_index = 0

        self.current_image = ...  # type: ReferenceImage
        self.current_comp = ...  # type: Component

        self.entropy_iter = iter([])  # type: Iterator[Component]

        self.bbox_count = 0
        self.ws_count = 0
        self.im_count = 0
        self.time_spent = 0
        
    def setupUI(self) -> None:
        self.setWindowTitle(self.tr("Window Name"))
        self.resize(800, 600)

        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        self.menubar = QMenuBar(self)
        self.setMenuBar(self.menubar)
        self.file_menu = self.menubar.addMenu(self.tr("&File"))

        self.help_menu = self.menubar.addMenu(self.tr("&Help"))

        self.layout = QHBoxLayout(centralWidget)

        self.proj_frame = ProjectionFrame(self)
        self.layout.addWidget(self.proj_frame)

        if Config.is_cityscapes():
            self.proj_frame.loadCityScapesCategoryLabels()

        self.im_frame = ImageFrame(self)
        self.layout.addWidget(self.im_frame)

        self.query_view = QListView(self)
        self.query_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.query_view.setViewMode(QListView.IconMode)
        self.query_view.setMaximumWidth(200)
        self.query_view.setIconSize(QSize(180, 180))
        self.query_view.setSelectionMode(QListView.NoSelection)
        self.layout.addWidget(self.query_view)  # Turned OFF

    def setupSignals(self) -> None:
        # ranking highlight
        self.proj_frame.view.border_nn_position.signal.connect(self.scrollToModelIndex)
        # im view watershed hierarchy evaluation
        self.im_frame.params_updated.connect(self.addProposedContourToDisplay)
        self.im_frame.hier_confirmed.connect(self.reloadWSHieararchy)
        # find component in current image
        self.im_frame.view.component_clicked.connect(self.findComponent)
        # seed watershed segm
        self.im_frame.view.markers_updated.connect(self.segmentByMarkers)
        self.im_frame.segm_confirmed.connect(self.confirmMarkerSegmentation)
        # label component from im view
        self.im_frame.view.label_position_clicked.connect(self.labelComponentAt)

    def setupActions(self) -> None:
        load_im_dir_act = QAction(self.tr("&Open Image Directory"), self.menubar)
        load_im_dir_act.setShortcut("Ctrl+O")
        load_im_dir_act.triggered.connect(self.openImageDirectory)
        self.file_menu.addAction(load_im_dir_act)

        load_bd_dir_act = QAction(self.tr("Open &Boundary Directory"), self.menubar)
        load_bd_dir_act.setShortcut("Ctrl+B")
        load_bd_dir_act.triggered.connect(self.openBoundaryDirectory)
        self.file_menu.addAction(load_bd_dir_act)

        commands_act = QAction(self.tr("Show UI Commands"), self.menubar)
        commands_act.triggered.connect(self.showCommandsDialog)
        self.help_menu.addAction(commands_act)

    @Slot()
    def showCommandsDialog(self) -> None:
        dialog = CommandsDialog(self)
        dialog.show()

    @Slot()
    def openImageDirectory(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, self.tr("Open Image Directory"),
                                                    QDir.homePath(),
                                                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if dir_path:
            self.im_dir = QDir(dir_path)
        self.proj_frame.proj_exec.setEnabled((self.im_dir is not None and self.bd_dir is not None))

    @Slot()
    def openBoundaryDirectory(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, self.tr("Open Boundary Directory"),
                                                    QDir.homePath(),
                                                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if dir_path:
            self.bd_dir = QDir(dir_path)
        self.proj_frame.proj_exec.setEnabled((self.im_dir is not None and self.bd_dir is not None))

    @Slot()
    def executeMainLoader(self) -> None:
        with wait_cursor():
            if hasattr(self, 'main_data') and hasattr(self, 'embedding'):
                self.embedding.load2dProj()
                self.proj_frame.view.loadImages(iter(self.main_data))
            else:
                print('Loading ... :D')
                tic = time.time()

                self.loader, self.main_data = get_mainloader(self.im_dir.absolutePath(),
                    self.bd_dir.absolutePath(), size=(224, 224), preload=Config.preload(),
                    pattern=Config.pattern(), sample=Config.sample(), batch_size=Config.batch_size(),
                    n_jobs=Config.n_jobs(),
                    gts_dir = Config.groundtruth_dir() if Config.use_gt() else None,
                )
                m_limit = Config.batch_load_size()
                self.main_data.current_limit = m_limit
                self.proj_frame.loading_limit.setValue(m_limit)
                
                elapsed = time.time() - tic
                print('Data loading time ...\t', time.strftime("%H:%M:%S", time.gmtime(elapsed)))
                tic = time.time()

                self.embedding = Embedding(self.loader, network=Config.embedding_network(),
                        cuda=Config.cuda(), weigths_path=Config.weights_path())

                elapsed = time.time() - tic
                print('Embeddingg time ...\t', time.strftime("%H:%M:%S", time.gmtime(elapsed)))
                tic = time.time()

                self.executeProjection()

                elapsed = time.time() - tic
                print('Projection time ...\t', time.strftime("%H:%M:%S", time.gmtime(elapsed)))

                self.init_time = time.time()
    
    @Slot()
    def executeMetricLearn(self) -> None:
        if not hasattr(self, 'embedding'):
            raise Warning('Something is wrong, metric learn was computed before embedding')
        
        with wait_cursor():
            self.proj_frame.view.resetSelection()
            balanced_data = get_balancedloader(self.main_data)
   
            ml.optimize_triplets(balanced_data, self.embedding.model, cuda=self.embedding.cuda)

            self.executeProjection()
            self.entropy_iter = iter([])  # resetting iterator

    @Slot()
    def setCurrent(self, comp: Component) -> None:
        self.current_image = comp.ref_image
        self.current_comp = comp

    def executeProjection(self, recompute_all: bool = True) -> None:
        self.embedding.loadNetworkFeatures(recompute_all)
        self.embedding.load2dProj()
        self.proj_frame.view.loadImages(iter(self.main_data))

    @Slot(list)
    def projectionPopUp(self, selected: List[Component]) -> None:
        self.sub_proj_frame = ProjSubsetFrame(self.proj_frame.label_box, self)
        self.sub_proj_frame.show()

        positions = self.embedding.load2dProjSubset(selected)
        self.sub_proj_frame.view.loadImages(iter(selected), positions)
        self.sub_proj_frame.view.is_visible = self.proj_frame.view.is_visible

    @Slot()
    def setCompDisplay(self, comp: Component) -> None:
        if self.current_image != comp.ref_image:
            self.setImageDisplay(comp.ref_image)
        self.setCurrent(comp)
        self.im_frame.view.appendComponent(ImageQt(comp.object_overlay()))

    @Slot()
    def setImageDisplay(self, image: ReferenceImage) -> None:
        self.current_image = image
        self.im_frame.label.setText(image.name)
        self.im_frame.view.addImage(ImageQt(self.current_image.image))
        self.im_frame.view.appendContour(ImageQt(cont.retrive_image_contour(self.current_image)))
        self.im_frame.altitude_box.setValue(self.current_image.ws_params['altitude'])
        self.im_frame.frontier_box.setValue(self.current_image.ws_params['frontier'])
        self.im_frame.hierarchy_box.setCurrentText(self.current_image.ws_params['hierarchy'])
        self.updateImageLabels()
        self.im_frame.setComboBoxValues(self.proj_frame.label_box)  # FIXME not very nice

        if hasattr(self, 'gt_mask') and self.current_image.gt is not None:
            red = mask_to_red(self.current_image.gt != 0)
            self.gt_view.setCurrent(ImageQt(self.current_image.image), red)

    @Slot(dict)
    def reloadWSHieararchy(self, params: dict):
        prev_components = self.current_image.components
        for comp in prev_components:
            self.proj_frame.view.removeComponent(comp)

        self.current_image.update_params(params)
        self.current_image.load_components()
        self.embedding.loadImageComponents(self.current_image)

        for comp in self.current_image:
            self.proj_frame.view.insertComponent(comp)
            for prev_comp in prev_components:
                if comp == prev_comp:
                    comp.label = prev_comp.label
                    break
                
        self.proj_frame.view.resetSelection()
        self.setImageDisplay(self.current_image)

    @Slot(dict)
    def addProposedContourToDisplay(self, params: dict):
        proposed_image = self.current_image.clone(params)
        proposed_image.load_components()
        p_image = cont.retrive_image_contour(proposed_image, color=(0, 255, 0, 255))
        new_contour = ImageQt(p_image)
        self.im_frame.view.appendProposal(new_contour)

    def updateImageLabels(self) -> None:
        color_map = {}
        for text, color in self.proj_frame.color_table.items():
            color_map[text] = color.toTuple()
        array = self.current_image.get_color_label(color_map)
        image = q2np.array2qimage(array)
        self.im_frame.view.appendLabel(image)

    @Slot(tuple)
    def labelComponentAt(self, position: Tuple[int, int]):
        self.findComponent(position)
        self.current_comp.label = self.proj_frame.label_box.currentText()
        self.proj_frame.view.updateColor([self.current_comp])
        self.updateImageLabels()

    @Slot(tuple)
    def segmentByMarkers(self, markers: Tuple[Tuple[Tuple[int, int], ...], ...], confirmed: bool = False) -> None:
        positives, negatives = markers
        # check if empty
        if not positives or not negatives:
            if confirmed:
                if positives:
                    self.current_comp.label = self.im_frame.lb_left_box.currentText()
                else:  # negatives
                    self.current_comp.label = self.im_frame.lb_right_box.currentText()

                self.proj_frame.view.updateColor([self.current_comp])
                self.updateImageLabels()
                self.im_count += 1

        else:
            segmentation = im_seg.ift(self.current_comp, positives, negatives)
            new_comp = self.current_comp.subset(segmentation)

            if not new_comp.empty() and new_comp != self.current_comp:
                if confirmed:
                    with wait_cursor():
                        self.splitCurrentComp(new_comp, self.im_frame.lb_left_box.currentText(),
                                              self.im_frame.lb_right_box.currentText())
                        self.ws_count += 1
                else:
                    self.im_frame.view.appendProposal(ImageQt(cont.retrive_comp_contour(new_comp)))

    @Slot(tuple)
    def confirmMarkerSegmentation(self, markers: Tuple[Tuple[int, int]]) -> None:
        self.segmentByMarkers(markers, True)
        self.proj_frame.view.resetSelection()

    def splitCurrentComp(self, new_comp: Component, new_label: str = 'void', current_label: str = 'void') -> None:
        diff_comp = self.current_comp - new_comp

        self.current_image.swap(self.current_comp, diff_comp)
        self.current_image.components.append(new_comp)

        new_comp.label = new_label
        diff_comp.label = current_label

        self.embedding.loadSingleComponent(diff_comp)
        self.embedding.loadSingleComponent(new_comp)

        self.proj_frame.view.removeComponent(self.current_comp)
        self.proj_frame.view.insertComponent(diff_comp)
        self.proj_frame.view.insertComponent(new_comp)

        self.setCompDisplay(diff_comp)
        self.im_frame.view.appendContour(ImageQt(cont.retrive_image_contour(self.current_image)))
        self.updateImageLabels()

    @Slot()
    def queryIndex(self, comp: Component) -> None:
        values, components = self.embedding.kClosests(comp)
        model = QStandardItemModel(len(components), 1, self.query_view)
        for i, (c, val) in enumerate(zip(components, values)):
            pixmap = QPixmap.fromImage(component_to_qimage(c))
            item = QStandardItem(QIcon(pixmap), "%.4f" % val)
            model.setItem(i, 0, item)
        self.query_view.setModel(model)

    @Slot(int)
    def scrollToModelIndex(self, index: int) -> None:
        for i in range(self.query_view.model().rowCount()):
            color = Qt.green if i < index else Qt.transparent
            item = self.query_view.model().item(i, 0)
            item.setData(QBrush(color), Qt.BackgroundRole)
        m_index = self.query_view.model().index(index, 0)
        self.query_view.scrollTo(m_index, self.query_view.PositionAtCenter)

    @Slot(tuple)
    def findComponent(self, pos: Tuple[int, int]) -> None:
        for comp in self.current_image.components:
            if comp.inside_component(x=pos[1], y=pos[0]):
                self.im_frame.view.appendComponent(ImageQt(comp.object_overlay()))
                self.current_comp = comp
                self.proj_frame.view.focusComponent(comp)
                break

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_D or event.key() == Qt.Key_A:
            self.im_index += 1 if event.key() == Qt.Key_D else -1
            if 0 <= self.im_index < len(self.main_data.reference_ims) and self.im_index < self.main_data.current_limit:
                self.current_image = self.main_data.reference_ims[self.im_index]
                self.current_comp = None
                self.setImageDisplay(self.current_image)
            elif self.im_index < 0:
                self.im_index = min(len(self.main_data.reference_ims), self.main_data.current_limit) - 1
            else:
                self.im_index = 0
        elif event.key() == Qt.Key_W:
            self.im_frame.opacity_box.setValue(min(1.0, self.im_frame.opacity_box.value() + 0.05))
        elif event.key() == Qt.Key_S:
            self.im_frame.opacity_box.setValue(max(0.0, self.im_frame.opacity_box.value() - 0.05))
        elif event.key() == Qt.Key_E:
            try:
                self.setCompDisplay(next(self.entropy_iter))
            except StopIteration:
                self.entropy_iter = self.embedding.kMaxEntropy(2.0)
                self.setCompDisplay(next(self.entropy_iter))

    @Slot(int)
    def setLoadingLimit(self, limit: int):
        self.main_data.current_limit = limit
        if hasattr(self, 'embedding'):
            with wait_cursor():
                self.executeProjection(False)

    @Slot()
    def unpickleData(self) -> None:
        with open(Config.pickle_path(), 'rb') as f:
            print('Unpickling ...')
            dic = pickle.load(f)
            if 'main_data' in dic:
                self.main_data = dic['main_data']
            if 'loader' in dic:
                self.loader = dic['loader']
            if 'embedding' in dic:
                self.embedding = dic['embedding']
                with wait_cursor():
                    self.embedding.load2dProj()

            self.bbox_count = dic['bbox_count']
            self.ws_count = dic['ws_count']
            self.im_count = dic['im_count']
            self.time_spent = dic['time_spent'] 
            self.init_time = time.time()

        if hasattr(self, 'main_data'):
            labels = self.main_data.labels()
            labels.discard('void')
            self.proj_frame.reloadLabelComboBox(labels)
            if hasattr(self, 'embedding'):
                self.proj_frame.view.loadImages(iter(self.main_data))
        print('Done!')

    @Slot()
    def pickleData(self) -> None:
        self.print_stats()

        dic = {}
        if hasattr(self, 'embedding'):
            backup_umap = self.embedding.umap
            del self.embedding.umap
            dic['embedding'] = self.embedding
        if hasattr(self, 'main_data'):
            dic['main_data'] = self.main_data
        if hasattr(self, 'loader'):
            dic['loader'] = self.loader
        
        dic['bbox_count'] = self.bbox_count + self.proj_frame.view.bbox_count
        dic['ws_count'] = self.ws_count
        dic['im_count'] = self.im_count
        dic['time_spent'] = self.time_spent + (time.time() - self.init_time)  
        self.init_time = time.time()

        with open(Config.pickle_path(), 'wb') as f:
            print('Pickling ...')
            pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)
            print('Done!')

        if hasattr(self, 'embedding'):
            self.embedding.umap = backup_umap

    @Slot()
    def saveSegmentation(self) -> None:
        ds_dir = os.path.join(Config.save_masks_path(), Config.current_dataset())
        os.makedirs(ds_dir, exist_ok=True)

        masks_dir = os.path.join(ds_dir, datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        os.makedirs(masks_dir)
        print('Saving masks in', masks_dir)

        label_map = {
            'void': 0,
        }
        if Config.is_cityscapes():
            label_map['ignored'] = 0
            for label in CS_LABELS:
                label_map[label.name] = label.id
        else:
            label_map['background'] = 0

        self.main_data.save_label_images(masks_dir, label_map, default_label=1)
        print('Done!')

    @Slot()
    def reload_all(self) -> None:
        importlib.reload(ml)
        importlib.reload(im_seg)
        importlib.reload(cont)

    @Slot()
    def print_stats(self) -> None:
        time_spent = time.strftime("%H:%M:%S", time.gmtime(self.time_spent + (time.time() - self.init_time)))
        bbox_count = self.bbox_count + self.proj_frame.view.bbox_count
        print(f'bbox count {bbox_count}\n' + 
              f'ws count {self.ws_count}\n' + 
              f'im count {self.im_count}\n' + 
              f'time {time_spent}\n')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(sys.argv)
    
    window.im_dir = QDir(Config.images_dir())
    window.bd_dir = QDir(Config.boundaries_dir())
    window.proj_frame.proj_exec.setEnabled(True)

    window.show()

    sys.exit(app.exec_())
