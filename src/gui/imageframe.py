from PySide2.QtCore import Qt, Slot, Signal
from PySide2.QtWidgets import (QFrame, QVBoxLayout, QGridLayout, QLabel, QSizePolicy,
                               QDoubleSpinBox, QPushButton, QComboBox)
from PySide2.QtGui import QKeyEvent

from .imageview import ImageView


class ImageFrame(QFrame):
    params_updated = Signal(dict)
    hier_confirmed = Signal(dict)
    segm_confirmed = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()
        self.setupSignals()

    def setupUI(self) -> None:
        self.layout = QVBoxLayout(self)
        self.label = QLabel(self)
        self.view = ImageView(self)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.view)

        min_policy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        but_frame = QFrame(self)
        but_layout = QGridLayout(but_frame)
        self.layout.addWidget(but_frame)

        # row 0
        descr_label = QLabel(self.tr("Left Click Label"))
        descr_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        descr_label.setSizePolicy(min_policy)
        but_layout.addWidget(descr_label, 0, 0)

        descr_label = QLabel(self.tr("Right Click Label"))
        descr_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        descr_label.setSizePolicy(min_policy)
        but_layout.addWidget(descr_label, 0, 1)

        descr_label = QLabel(self.tr("Labels Opacity"))
        descr_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        descr_label.setSizePolicy(min_policy)
        but_layout.addWidget(descr_label, 0, 3)

        # row 1
        self.lb_left_box = QComboBox(self)
        self.lb_left_box.setSizePolicy(min_policy)
        self.lb_left_box.setEditable(False)
        but_layout.addWidget(self.lb_left_box, 1, 0)

        self.lb_right_box = QComboBox(self)
        self.lb_right_box.setSizePolicy(min_policy)
        self.lb_right_box.setEditable(False)
        but_layout.addWidget(self.lb_right_box, 1, 1)

        self.confirm_seg_but = QPushButton(self.tr("Split"), self)
        self.confirm_seg_but.setSizePolicy(min_policy)
        but_layout.addWidget(self.confirm_seg_but, 1, 2)

        self.opacity_box = QDoubleSpinBox(self)
        self.opacity_box.setRange(0, 1)
        self.opacity_box.setValue(0.25)
        self.opacity_box.setSingleStep(0.05)
        self.opacity_box.setSizePolicy(min_policy)
        but_layout.addWidget(self.opacity_box, 1, 3)
        self.view.label_opacity.setOpacity(self.opacity_box.value())

        # row 2
        descr_label = QLabel(self.tr("WS Contour Filter"))
        descr_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        descr_label.setSizePolicy(min_policy)
        but_layout.addWidget(descr_label, 2, 0)

        descr_label = QLabel(self.tr("WS Volume Threshold"))
        descr_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        descr_label.setSizePolicy(min_policy)
        but_layout.addWidget(descr_label, 2, 1)

        self.update_but = QPushButton(self.tr("WS Preview"), self)
        self.update_but.setSizePolicy(min_policy)
        but_layout.addWidget(self.update_but, 2, 2)

        self.confirm_hier_but = QPushButton(self.tr("WS Recompute"), self)
        self.confirm_hier_but.setSizePolicy(min_policy)
        but_layout.addWidget(self.confirm_hier_but, 2, 3)

        # row 3
        self.frontier_box = QDoubleSpinBox(self)
        self.frontier_box.setRange(0.05, 0.95)
        self.frontier_box.setValue(0.05)
        self.frontier_box.setSingleStep(0.05)
        self.frontier_box.setSizePolicy(min_policy)
        but_layout.addWidget(self.frontier_box, 3, 0)

        self.altitude_box = QDoubleSpinBox(self)
        self.altitude_box.setRange(50, 20000)
        self.altitude_box.setValue(100)
        self.altitude_box.setSingleStep(50)
        self.altitude_box.setSizePolicy(min_policy)
        but_layout.addWidget(self.altitude_box, 3, 1)

        self.hierarchy_box = QComboBox(self)
        self.hierarchy_box.setSizePolicy(min_policy)
        self.hierarchy_box.addItems(('area', 'dynamics', 'volume'))
        but_layout.addWidget(self.hierarchy_box, 3, 2)

        
    def setupSignals(self) -> None:
        # parameters update
        self.update_but.clicked.connect(self.updated)
        # confirming segemtantion
        self.confirm_seg_but.clicked.connect(self.segmentationConfirmed)
        # reload hierarchy
        self.confirm_hier_but.clicked.connect(self.reloadHierarchy)
        # label alpha update
        self.opacity_box.valueChanged.connect(self.view.label_opacity.setOpacity)

    @Slot()
    def updated(self) -> None:
        params = {'altitude': self.altitude_box.value(),
                  'frontier': self.frontier_box.value(),
                  'hierarchy': self.hierarchy_box.currentText(),
                  }
        self.params_updated.emit(params)

    @Slot()
    def reloadHierarchy(self) -> None:
        params = {'altitude': self.altitude_box.value(),
                  'frontier': self.frontier_box.value(),
                  'hierarchy': self.hierarchy_box.currentText(),
                }
        self.hier_confirmed.emit(params)

    @Slot()
    def segmentationConfirmed(self) -> None:
        self.segm_confirmed.emit((self.view.positives_markers, self.view.negatives_markers))
        self.view.clearMarkers()

    def setComboBoxValues(self, box: QComboBox):
        prev_left = self.lb_left_box.currentText()
        prev_right = self.lb_right_box.currentText()
        self.lb_left_box.clear()
        self.lb_right_box.clear()
        for i in range(box.count()):
            icon = box.itemIcon(i)
            text = box.itemText(i)
            self.lb_left_box.addItem(icon, text)
            self.lb_right_box.addItem(icon, text)
        self.lb_left_box.setCurrentText('void' if not prev_left else prev_left)
        self.lb_right_box.setCurrentText('void' if not prev_right else prev_right)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Space:
            self.segmentationConfirmed()
