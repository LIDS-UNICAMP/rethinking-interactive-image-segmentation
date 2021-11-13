import random
from typing import List

from PySide2.QtCore import Qt, Slot, Signal
from PySide2.QtWidgets import (QFrame, QPushButton, QComboBox, QSpinBox,
                               QVBoxLayout, QGridLayout, QColorDialog, QLabel,
                               QSizePolicy, QCheckBox, )
from PySide2.QtGui import QPixmap, QIcon, QColor

from .projectionview import ProjectionView

from loaders.labelcityscape import labels as CS_LABELS


class ProjectionFrame(QFrame):
    pen_color_changed = Signal(QColor)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()
        self.setupSignals()

        self.color_table = {'void': QColor(0, 0, 0, 0)}

    def setupUI(self) -> None:
        min_policy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        fixed_policy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.layout = QVBoxLayout(self)
        self.view = ProjectionView(self)
        self.layout.addWidget(self.view)

        lower_frame = QFrame(self)
        self.layout.addWidget(lower_frame)
        lower_frame_layout = QGridLayout(lower_frame)
        lower_frame_layout.setAlignment(Qt.AlignLeft)

        # Row 0
        descr_label = QLabel(self.tr("Current Label"))
        descr_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        descr_label.setSizePolicy(min_policy)
        lower_frame_layout.addWidget(descr_label, 0, 0)

        self.label_box = QComboBox(lower_frame)
        self.label_box.setEditable(True)
        self.label_box.setSizePolicy(min_policy)
        self.label_box.setInsertPolicy(QComboBox.InsertAtTop)
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.white)
        self.label_box.addItem(QIcon(pixmap), 'void')
        lower_frame_layout.addWidget(self.label_box, 0, 1)

        self.label_visible = QCheckBox(self.tr("Labeled Invisible"), lower_frame)
        self.label_visible.setSizePolicy(min_policy)
        lower_frame_layout.addWidget(self.label_visible, 0, 2)

        self.subset_select = QCheckBox(self.tr("Select Subset"), lower_frame)
        self.subset_select.setSizePolicy(min_policy)
        lower_frame_layout.addWidget(self.subset_select, 0, 3)

        # Row 1
        descr_label = QLabel(self.tr("Confirm Selection"))
        descr_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        descr_label.setSizePolicy(min_policy)
        lower_frame_layout.addWidget(descr_label, 1, 0)

        self.label_confirm = QPushButton(self.tr("Confirm"), lower_frame)
        self.label_confirm.setEnabled(False)
        self.label_confirm.setSizePolicy(min_policy)
        lower_frame_layout.addWidget(self.label_confirm, 1, 1)

        self.label_cancel = QPushButton(self.tr("Cancel"), lower_frame)
        self.label_cancel.setEnabled(False)
        self.label_cancel.setSizePolicy(min_policy)
        lower_frame_layout.addWidget(self.label_cancel, 1, 2)

        self.save_segm = QPushButton(self.tr("Save Masks"), lower_frame)
        self.save_segm.setSizePolicy(min_policy)
        lower_frame_layout.addWidget(self.save_segm, 1, 3)

        self.loading_limit = QSpinBox(lower_frame)
        self.loading_limit.setKeyboardTracking(False)
        self.loading_limit.setSingleStep(10)
        self.loading_limit.setRange(10, int(1e5))
        self.loading_limit.setValue(10)
        self.loading_limit.setSizePolicy(min_policy)
        lower_frame_layout.addWidget(self.loading_limit, 1, 4)

        # Row 2
        self.proj_exec = QPushButton(self.tr("Execute Projection"), lower_frame)
        self.proj_exec.setEnabled(False)
        self.proj_exec.setSizePolicy(fixed_policy)
        lower_frame_layout.addWidget(self.proj_exec, 2, 0)

        self.ml_exec = QPushButton(self.tr("Execute Metric Learn"), lower_frame)
        self.ml_exec.setSizePolicy(fixed_policy)
        lower_frame_layout.addWidget(self.ml_exec, 2, 1)

        self.unpickle_but = QPushButton(self.tr('UNPICKLE!'), lower_frame)
        self.unpickle_but.setSizePolicy(min_policy)
        lower_frame_layout.addWidget(self.unpickle_but, 2, 2)

        self.pickle_but = QPushButton(self.tr('Save (Pickle)'), lower_frame)
        self.pickle_but.setSizePolicy(min_policy)
        lower_frame_layout.addWidget(self.pickle_but, 2, 3)

        self.reload_but = QPushButton(self.tr("Reload Modules"), lower_frame)
        self.reload_but.setSizePolicy(min_policy)
        lower_frame_layout.addWidget(self.reload_but, 2, 4)

    def setupSignals(self) -> None:
        # pen color
        self.pen_color_changed.connect(self.view.setPenColor)

        # proj view
        self.view.subset_selected.signal.connect(self.parent().projectionPopUp)
        self.view.item_clicked_left.signal.connect(self.parent().setCompDisplay)
        self.view.item_clicked_right.signal.connect(self.parent().queryIndex)
        self.view.is_selected.signal.connect(self.label_confirm.setEnabled)
        self.view.is_selected.signal.connect(self.label_cancel.setEnabled)

        # pickling
        self.unpickle_but.clicked.connect(self.parent().unpickleData)
        self.pickle_but.clicked.connect(self.parent().pickleData)

        # saving
        self.save_segm.clicked.connect(self.parent().saveSegmentation)

        # labels
        self.label_box.activated.connect(self.addLabelIcon)
        self.label_box.currentTextChanged.connect(self.view.resetSelection)
        self.label_cancel.clicked.connect(self.view.resetSelection)
        self.label_confirm.clicked.connect(self.view.confirmSelection)
        self.label_visible.toggled.connect(self.view.setLabeledInvisible)

        # projection and metric learning
        self.proj_exec.clicked.connect(self.parent().executeMainLoader)
        self.ml_exec.clicked.connect(self.parent().executeMetricLearn)

        # proj subset
        self.subset_select.toggled.connect(self.view.setSubsetSelection)
        self.subset_select.toggled.connect(self.view.resetSelection)

        # reloading
        self.reload_but.clicked.connect(self.parent().reload_all)

        # loading limit
        self.loading_limit.valueChanged.connect(self.parent().setLoadingLimit)


    def getIcon(self, color: QColor) -> QIcon:
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        return QIcon(pixmap)

    @Slot(str)
    def emitPenColor(self, text: str) -> None:
        color = self.color_table[text]
        self.pen_color_changed.emit(color)

    @Slot()
    def addLabelIcon(self) -> None:
        index = self.label_box.currentIndex()
        label = self.label_box.itemText(index)
        if label not in self.color_table:
            rand_color = QColor(*(random.randint(0, 255) for c in range(3)))
            color_dialog = QColorDialog(rand_color, self)
            color = color_dialog.getColor(rand_color)
            self.label_box.setItemIcon(index, self.getIcon(color))
            self.color_table[label] = color
        self.emitPenColor(label)
        self.subset_select.setChecked(False)
        self.view.setSubsetSelection(False)

    def loadCityScapesCategoryLabels(self):
        for label in CS_LABELS:
            if label.ignoreInEval:
                continue
            color = QColor(*label.color)
            self.label_box.addItem(self.getIcon(color), label.name)
            self.color_table[label.name] = color

        black = QColor(0, 0, 0)
        self.label_box.addItem(self.getIcon(black), 'ignored')
        self.color_table['ignored'] = black

        self.label_box.setEditable(False)
        self.label_box.activated.disconnect(self.addLabelIcon)
        self.label_box.currentTextChanged.connect(self.emitPenColor)

    def reloadLabelComboBox(self, labels: List[str]) -> None:
        for label in labels:
            if label in self.color_table.keys():
                continue
            rand_color = QColor(*(random.randint(0, 255) for c in range(3)))
            self.label_box.addItem(self.getIcon(rand_color), label)
            self.color_table[label] = rand_color
