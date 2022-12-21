# -*- coding: utf-8 -*-

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
                            QMetaObject, QObject, QPoint, QRect,
                            QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
                           QCursor, QFont, QFontDatabase, QGradient,
                           QIcon, QImage, QKeySequence, QLinearGradient,
                           QPainter, QPalette, QPixmap, QRadialGradient,
                           QTransform, QImageReader)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
                               QMenu, QMenuBar, QPushButton, QSizePolicy,
                               QStatusBar, QVBoxLayout, QWidget, QFileDialog, QDialog)
import resource
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np


def get_supported_mime_types():  # 지원하는 이미지 타입을 알려주는 함수
    result = []
    for f in QImageReader().supportedMimeTypes():
        data = f.data()
        result.append(data.decode('utf-8'))
    return result


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()

        # 이미지 경로를 저장하는 변수 초기화
        self.img_url = None

        # 마우스 위치 저장 변수 (사각형 그리기 위해서)
        self.begin = QPoint()
        self.end = QPoint()
        # 그리기 위한 도구
        self.painter = QPainter()
        # 브러쉬 설정
        self.br = QBrush(QColor(230, 10, 10, 75))  # R G B Alpha
        self.painter.setBrush(self.br)
        # 버튼을 눌렀을 때만 그릴 수 있도록 bool 변수 선언
        self.is_paint = False
        # 사각 박스 저장 리스트
        self.bbox_list = []
        # pixmap 초기화
        self.pixmap = None
        # pixmap 시작 좌표
        self.set_p_x = 0
        self.set_p_y = 0
        # 되돌아가는 command Ctrl+Z 등록
        prev_action = QAction(self)
        # prev_action.setShortcut(QKeySequence.Undo)  # 혹은
        prev_action.setShortcut(QKeySequence(Qt.CTRL | Qt.Key_Z))
        prev_action.triggered.connect(self.prev_bbox)
        self.addAction(prev_action)

        self.open_path = None

    def setupUi(self):
        if not self.objectName():
            self.setObjectName(u"MainWindow")
        self.resize(1280, 800)
        self.setMinimumSize(QSize(1280, 800))
        self.setMaximumSize(QSize(1280, 800))
        self.setSizeIncrement(QSize(1280, 800))
        self.setBaseSize(QSize(1280, 800))
        self.open = QAction(self)
        self.open.setObjectName(u"open")
        self.open.triggered.connect(self.img_open)

        self.save = QAction(self)
        self.save.setObjectName(u"save")
        self.save.triggered.connect(self.save_ann)

        self.close = QAction(self)
        self.close.setObjectName(u"close")
        self.close.triggered.connect(QCoreApplication.quit)

        self.action = QAction(self)
        self.action.setObjectName(u"action")
        self.action_2 = QAction(self)
        self.action_2.setObjectName(u"action_2")
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.paint = QPushButton(self.centralwidget)
        self.paint.setObjectName(u"paint")
        self.paint.setMinimumSize(QSize(0, 50))
        icon = QIcon()
        icon.addFile(u":/ann/draw.png", QSize(), QIcon.Normal, QIcon.Off)
        self.paint.setIcon(icon)
        self.paint.setIconSize(QSize(25, 25))
        self.paint.clicked.connect(self.clicked_paint)
        self.paint.setShortcut(QCoreApplication.translate("MainWindow", u"N", None))

        self.verticalLayout_2.addWidget(self.paint)

        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.img_label = QLabel(self.centralwidget)
        self.img_label.setObjectName(u"img_label")
        self.img_label.setAlignment(Qt.AlignCenter)  # 중앙배열

        self.horizontalLayout.addWidget(self.img_label)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 23)
        self.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(self)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1280, 30))
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        self.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName(u"statusbar")
        self.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menu.menuAction())
        self.menu.addAction(self.open)
        self.menu.addAction(self.save)
        self.menu.addAction(self.close)

        self.retranslateUi()

        QMetaObject.connectSlotsByName(self)

    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.open.setText(QCoreApplication.translate("MainWindow", u"\uc5f4\uae30", None))
        # if QT_CONFIG(shortcut)
        self.open.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+O", None))
        # endif // QT_CONFIG(shortcut)
        self.save.setText(QCoreApplication.translate("MainWindow", u"\uc800\uc7a5", None))
        # if QT_CONFIG(shortcut)
        self.save.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+S", None))
        # endif // QT_CONFIG(shortcut)
        self.close.setText(QCoreApplication.translate("MainWindow", u"\ub2eb\uae30", None))
        # if QT_CONFIG(shortcut)
        self.close.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+F4", None))
        # endif // QT_CONFIG(shortcut)
        self.action.setText(QCoreApplication.translate("MainWindow", u"\ubcf5\uc0ac", None))
        self.action_2.setText(QCoreApplication.translate("MainWindow", u"\ubd99\uc5ec\ub123\uae30", None))
        self.paint.setText("")
        self.img_label.setText("")
        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"\ud30c\uc77c", None))

    # retranslateUi

    def img_open(self):
        open_path = str(Path(__file__).resolve().parent)
        file_dialog = QFileDialog(self, '이미지 열기', open_path)
        self.open_path = open_path

        _mime_types = get_supported_mime_types()
        file_dialog.setMimeTypeFilters(_mime_types)

        default_mimetype = 'image/png'
        if default_mimetype in _mime_types:
            file_dialog.selectMimeTypeFilter(default_mimetype)

        if file_dialog.exec() == QDialog.Accepted:
            url = file_dialog.selectedUrls()[0]
            self.img_url = url
            print(url)

            self.pixmap = QPixmap(url.toLocalFile())  # localFile로 변환해야 돌아갈 때가 있다
            self.pixmap = self.pixmap.scaled(self.img_label.size(), aspectMode=Qt.KeepAspectRatio)  # 비율 유지한채 사이즈변경
            # self.img_label.setPixmap(pixmap)
            l_width = self.img_label.geometry().width()  # label크기의 x, y, w, h가 다 있음
            l_height = self.img_label.geometry().height()
            l_x = self.img_label.geometry().x()
            l_y = self.img_label.geometry().y()

            # 메뉴바 크기도 알아야됨, 0,0은 메뉴바에 가림
            m_height = self.menubar.geometry().height()

            l_center_x = (l_width // 2) + l_x
            l_center_y = (l_height // 2) + l_y + m_height

            p_width = self.pixmap.rect().width()
            p_height = self.pixmap.rect().height()

            self.set_p_x = l_center_x - (p_width // 2)
            self.set_p_y = l_center_y - (p_height // 2)

    # 버튼 눌렀을 때의 이벤트
    def clicked_paint(self):
        if not self.is_paint:
            self.is_paint = True

    def prev_bbox(self):
        # 맨 마지막 것을 제외한 나머지를 다시 그려야 함
        if len(self.bbox_list) == 0:
            return
        # 새로 그리기 위해서 pixmap을 다시 정의
        self.pixmap = QPixmap(self.img_url.toLocalFile())
        self.pixmap = self.pixmap.scaled(self.img_label.size(), aspectMode=Qt.KeepAspectRatio)

        self.bbox_list = self.bbox_list[:-1]

        self.painter.begin(self.pixmap)
        self.painter.setBrush(self.br)
        # 그리기전 시작과 끝을 항상 지정해야함
        for qrect in self.bbox_list:
            self.painter.drawRect(qrect)

        self.painter.end()
        self.update()

    def paintEvent(self, event):
        if self.pixmap:
            with QPainter(self) as painter:
                painter.drawPixmap(self.set_p_x, self.set_p_y, self.pixmap)  # 그릴 좌표 x, y

                if self.is_paint:
                    painter.setBrush(self.br)
                    painter.drawRect(QRect(self.begin, self.end))
                    # Q painter가 self를 받기 때문에 상대좌표가 아니어도 적용이 됨

    def mousePressEvent(self, event):
        self.begin = event.position().toPoint()  # 지금 내가 누른 좌표를 넣어줌
        QWidget.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        self.end = event.position().toPoint()  # 움직이는 좌표 계속 저장
        self.update()  # 움직이는 좌표에 따라 그림 미리보기 되게
        QWidget.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        self.end = event.position().toPoint()

        if self.pixmap and self.is_paint:  # pixmap이 존재할 때만 그림
            self.painter.begin(self.pixmap)
            self.painter.setBrush(self.br)
            begin = list(self.begin.toTuple())
            end = list(self.end.toTuple())
            begin[0] -= self.set_p_x
            begin[1] -= self.set_p_y
            end[0] -= self.set_p_x
            end[1] -= self.set_p_y

            w = end[0] - begin[0]
            h = end[1] - begin[1]
            self.painter.drawRect(QRect(begin[0], begin[1], w, h))
            # 상대좌표로 그리기 때문에 절대좌표를 상대좌표(이미지 상 좌표)로 바꿔서 넣어야함
            self.painter.end()

            self.bbox_list.append(QRect(begin[0], begin[1], w, h))
            # 좌표 리스트에 저장
            self.is_paint = False
        self.update()
        QWidget.mouseReleaseEvent(self, event)

    def save_ann(self):
        title = os.path.basename(self.img_url.toLocalFile())
        # print(title)
        # print(os.path.basename(self.img_url.toLocalFile()))
        arr_temp = np.zeros_like([i for i in range(len(self.bbox_list))]).reshape((-1, 1))
        arr_temp = pd.DataFrame(arr_temp, columns=['name'])
        arr_temp['name'] = arr_temp['name'].astype(str)
        arr_temp.iloc[:, 0] = title
        # print(arr_temp)

        bbox_temp = pd.DataFrame(self.bbox_list, columns=['bbox'])
        temp_csv = pd.concat([arr_temp, bbox_temp], axis=1)
        csv_ann = temp_csv.reset_index()
        csv_ann = csv_ann.iloc[:, 1:]
        print(csv_ann)
        csv_ann.to_csv('./data_ann.csv')

    def closeEvent(self, close_event):
        self.deleteLater()
        close_event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # app.setStyle('Fusion')
    widget = Ui_MainWindow()
    widget.show()
    sys.exit(app.exec())
