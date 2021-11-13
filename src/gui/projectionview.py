from PySide2.QtCore import Qt, Slot, Signal, QRectF, QObject, QPointF
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsColorizeEffect
from PySide2.QtGui import QMouseEvent, QPixmap, QWheelEvent, QPen, QColor, QTransform

from .utils import component_to_qimage, DefaultOrderedDict
from loaders.maindata import Component

import numpy as np
from typing import List, Tuple, Iterator


class IsSelectedSignal(QObject):
    signal = Signal(bool)


class ClickedItemSignal(QObject):
    signal = Signal(Component)


class SubsetSignal(QObject):
    signal = Signal(list)


class SelectionSignal(QObject):
    signal = Signal(tuple)


class DragDistSignal(QObject):
    signal = Signal(float)


class CurrentBorderIndex(QObject):
    signal = Signal(int)


class ProjectionView(QGraphicsView):
    def __init__(self, parent=None, scale=500):
        super().__init__(parent)

        self.setTransformationAnchor(self.NoAnchor)
        self.setResizeAnchor(self.NoAnchor)
        
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        # avoid crashing; bug issue: https://bugreports.qt.io/browse/QTBUG-18021
        self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)

        self.proj_scale = scale    
        self.graphitem_to_comp = {}
        self.comp_to_graphitem = {}
        self.selection = []
        self.prev_effect_color = {}

        self.view_scale = 1.0

        self.prev_pos = None
        self.cur_rect = None
        self.cur_pen = QPen(Qt.white, 10, cap=Qt.RoundCap)
        self.cur_color = None
        self.effect_str = 0.65

        self.ordered_neigh = DefaultOrderedDict(list)
        self.prev_threshold = -1

        self.is_visible = True
        self.proj_subset = False
    
        self.bbox_count = 0
        self.prev_z = 0
        self.setupSignals()

    def setupSignals(self) -> None:
        self.is_selected = IsSelectedSignal(self)
        self.subset_selected = SubsetSignal(self)
        self.item_clicked_left = ClickedItemSignal(self)
        self.item_clicked_right = ClickedItemSignal(self)

        # knn selection over mouse movement
        self.drag_dist_moved = DragDistSignal(self)
        self.drag_dist_moved.signal.connect(self.selectWithinThreshold)

        # qgraphics item colorization
        self.select_this = SelectionSignal(self)
        self.select_this.signal.connect(self.appendAndColorizeSet)
        self.reset_this = SubsetSignal(self)
        self.reset_this.signal.connect(self.decolorizeSet)

        self.border_nn_position = CurrentBorderIndex(self)

    def resetCanvas(self) -> None:
        self.graphitem_to_comp.clear()
        self.comp_to_graphitem.clear()
        self.scene.clear()

    def loadImages(self, components: Iterator[Component], positions: np.ndarray = None) -> None:
        self.resetCanvas()

        min_x, max_x, min_y, max_y = 1e23, -1e23, 1e23, -1e23
        for idx, comp in enumerate(components):
            if positions is None:
                item = self.drawComponent(comp)
            else:
                item = self.drawComponent(comp, position=positions[idx])
            self.graphitem_to_comp[item] = comp
            self.comp_to_graphitem[comp] = item
            min_x = min(min_x, comp.position[0])
            max_x = max(max_x, comp.position[0])
            min_y = min(min_y, comp.position[1])
            max_y = max(max_y, comp.position[1])
        self.fitInView((min_x, max_x, min_y, max_y))

        if not self.is_visible:
            self.setLabeledInvisible(True)

    def fitInView(self, boundary: Tuple[float, float, float, float], margin: int=2) -> None:
        m_x, M_x = boundary[0] - margin, boundary[1] + margin
        m_y, M_y = boundary[2] - margin, boundary[3] + margin
        m_x, M_x = m_x * self.proj_scale, M_x * self.proj_scale
        m_y, M_y = m_y * self.proj_scale, M_y * self.proj_scale
        self.view_scale = 1.0
        super().fitInView(m_x, m_y, M_x - m_x, M_y - m_y)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.prev_pos = self.mapToScene(event.pos())
        if event.button() is Qt.LeftButton:
            self.cur_rect = self.scene.addRect(QRectF(self.prev_pos, self.prev_pos), self.cur_pen)
        elif event.button() is Qt.RightButton:
            item = self.scene.itemAt(self.prev_pos, QTransform())
            if item is not None:
                self.item_clicked_right.signal.emit(self.graphitem_to_comp[item])
                self.ordered_neigh = self.queryNeighbours(self.graphitem_to_comp[item])

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.prev_pos is None:
            return
        if self.cur_rect is not None:
            cur_pos = self.mapToScene(event.pos())
            self.cur_rect.setRect(QRectF(self.prev_pos, cur_pos))
        elif len(self.ordered_neigh):
            diff = self.prev_pos - self.mapToScene(event.pos())
            self.drag_dist_moved.signal.emit(self.dragDistanceFunction(diff))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.prev_pos is None:
            return
        if event.button() is Qt.LeftButton and self.cur_rect is not None:
            candidates = self.cur_rect.collidingItems()
            # resetting rect
            self.scene.removeItem(self.cur_rect)
            self.cur_rect = None
            # filtering invisibles
            selection = [item for item in candidates if item.isVisible()]
            if selection:
                # picking color
                color = self.cur_color if not self.proj_subset else QColor(255, 0, 127)
                self.select_this.signal.emit((color, selection))
        elif event.button() is Qt.RightButton and len(self.ordered_neigh):
            self.ordered_neigh.clear()
            self.prev_threshold = -1
        self.prev_pos = None 

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() == Qt.CTRL:
            old_pos = self.mapToScene(event.pos())
            in_factor = 1.1
            out_factor = 0.9
            factor = 1.0
            # y_delta = event.pixelDelta().y()  # macOS only
            y_delta = event.angleDelta().y()
            if y_delta > 0:
                factor = in_factor
            elif y_delta < 0:
                factor = out_factor
            self.view_scale *= factor
            self.scale(factor, factor)
            new_pos = self.mapToScene(event.pos())
            delta = new_pos - old_pos
            self.translate(delta.x(), delta.y())
        else:
            scroll_scale = 1
            # shift = event.pixelDelta()  # macOS only
            shift = event.angleDelta()
            self.translate(scroll_scale * shift.x(),
                           scroll_scale * shift.y())

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        item = self.scene.itemAt(self.mapToScene(event.pos()), QTransform())
        if item is not None:
            if event.button() is Qt.LeftButton:
                self.item_clicked_left.signal.emit(self.graphitem_to_comp[item])
            elif event.button() is Qt.RightButton:
                self.item_clicked_right.signal.emit(self.graphitem_to_comp[item])

    @Slot()
    def resetSelection(self) -> None:
        for item in self.selection:
            self.colorizeItem(item, self.prev_effect_color[item])
        self.selection.clear()
        self.prev_effect_color.clear()
        self.is_selected.signal.emit(False)

    @Slot()
    def confirmSelection(self) -> None:
        if self.proj_subset:
            subset = [0] * len(self.selection)
            for i, item in enumerate(self.selection):
                subset[i] = self.graphitem_to_comp[item]
            self.resetSelection()
            self.subset_selected.signal.emit(subset)
        else:
            cur_label = self.parent().label_box.currentText()
            for item in self.selection:
                self.graphitem_to_comp[item].label = cur_label
                if cur_label != 'void':  # check if is not void
                    item.setVisible(self.is_visible)
            self.selection.clear()
            self.prev_effect_color.clear()
            self.is_selected.signal.emit(False)

            self.bbox_count += 1

    @Slot(QColor)
    def setPenColor(self, color: QColor) -> None:
        self.cur_pen.setColor(color)
        self.cur_color = color

    def colorizeItem(self, item: QGraphicsItem, color: QColor) -> None:
        if color is None:
            return
        effect = QGraphicsColorizeEffect(self.scene)
        effect.setColor(color)
        effect.setStrength(self.effect_str * color.alphaF())
        item.setGraphicsEffect(effect)

    @Slot(bool)
    def setLabeledInvisible(self, status: bool) -> None:
        self.is_visible = not status
        for item, comp in self.graphitem_to_comp.items():
            if comp.label != 'void':
                item.setVisible(self.is_visible)

    @Slot(bool)
    def setSubsetSelection(self, status: bool) -> None:
        self.proj_subset = status

    @Slot(tuple)
    def appendAndColorizeSet(self, args: Tuple[QColor, List[QGraphicsItem]]) -> None:
        color, items = args
        for item in items:
            if item not in self.prev_effect_color:
                eff = item.graphicsEffect()
                self.prev_effect_color[item] = eff.color() if eff else None
            self.colorizeItem(item, color)
        
        self.selection += items
        if self.selection:
            self.is_selected.signal.emit(True)

    @Slot(list)
    def decolorizeSet(self, items: List[QGraphicsItem]) -> None:
        for item in items:
            self.colorizeItem(item, self.prev_effect_color[item])

    def queryNeighbours(self, comp: Component, kmax: int=500) -> DefaultOrderedDict:
        pqueue = DefaultOrderedDict(list)
        main_window = self.parent().parent().parent() # TODO change everything to QApplication
        values, comps = main_window.embedding.kClosests(comp, only='void', k=kmax)
        for v, c in zip(values, comps):
            pqueue[v].append(c)
        return pqueue

    def dragDistanceFunction(self, point: QPointF) -> float:
        scale = 1e3
        dist = np.sqrt(QPointF.dotProduct(point, point))
        dist = dist / scale
        return dist

    @Slot(float)
    def selectWithinThreshold(self, threshold: float) -> None:
        if not len(self.ordered_neigh):
            return
        selected = []
        to_be_cleared = []
        count = 0
        for v, it_list in self.ordered_neigh.items():
            if (1 - v) <= threshold:
                selected += [self.comp_to_graphitem[comp] for comp in it_list]
                count += len(it_list)
            elif (1 - v) <= self.prev_threshold:
                to_be_cleared += [self.comp_to_graphitem[comp] for comp in it_list]
            else:
                break
        
        self.border_nn_position.signal.emit(count)
        self.prev_threshold = threshold
        self.selection.clear()
        self.reset_this.signal.emit(to_be_cleared)
        self.select_this.signal.emit((self.cur_color, selected))

    @Slot(list)
    def updateColor(self, components: List[Component]) -> None:
        for comp in components:
            item = self.comp_to_graphitem[comp]
            self.colorizeItem(item, self.parent().color_table[comp.label])
            if comp.label != 'void':
                item.setVisible(self.is_visible)

    def drawComponent(self, comp: Component, position: np.ndarray = None) -> QGraphicsItem:
        pixmap = QPixmap.fromImage(component_to_qimage(comp))
        if pixmap.width() > 320 or pixmap.height() > 320:
            pixmap = pixmap.scaled(320, 320, Qt.KeepAspectRatio)
        item = self.scene.addPixmap(pixmap)
        if position is None:
            position = comp.position
        item.setPos(self.proj_scale * position[0], self.proj_scale * position[1])
        self.colorizeItem(item, self.parent().color_table[comp.label])
        if comp.label != 'void':
            item.setVisible(self.is_visible)
        return item

    def removeComponent(self, comp: Component) -> None:
        item = self.comp_to_graphitem[comp]
        # bug thread: https://forum.qt.io/topic/75510/deleting-qgraphicsitem-with-qgraphicseffect-leads-to-segfault
        # bug issue: https://bugreports.qt.io/browse/QTBUG-18021
        item.prepareGeometryChange()  # workaround, updates Binary Space Partitioning index
        self.scene.removeItem(item)
        del self.comp_to_graphitem[comp]
        del self.graphitem_to_comp[item]

    def insertComponent(self, comp: Component) -> None:
        item = self.drawComponent(comp)
        self.comp_to_graphitem[comp] = item
        self.graphitem_to_comp[item] = comp

    def focusComponent(self, comp: Component) -> None:
        item = self.comp_to_graphitem[comp]
        if item.isVisible():
            item.ensureVisible()
            self.prev_z += 1
            item.setZValue(self.prev_z)
