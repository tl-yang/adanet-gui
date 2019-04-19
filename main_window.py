from abc import ABCMeta

from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject, QCoreApplication, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QFileDialog

from ui_main_widget import Ui_MainWidget


class FinalMeta(ABCMeta, type(QObject)):
    pass


class MainWindow(QWidget, metaclass=FinalMeta):
    train_btn_click_signal = pyqtSignal()
    test_btn_click_signal = pyqtSignal()
    dataset_select_signal = pyqtSignal(str)
    load_custom_test_signal = pyqtSignal(str)
    use_default_test_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        ui = Ui_MainWidget()
        ui.setupUi(self)
        self.ui = ui
        self._init()
        self._connect()

    def append_text_in_textedit(self, text):
        self.ui.textEdit.append(text)

    def _init(self):
        self.ui.Dataset.model().item(0).setEnabled(False)
        self.ui.textEdit.setReadOnly(True)
        self.ui.LoadCustomTest.setEnabled(False)
        self.ui.UseDefaultTest.setEnabled(False)
        self.ui.TestBtn.setEnabled(False)
        self.ui.TrainBtn.setEnabled(True)

    def _connect(self):
        # self.ui
        self.ui.TrainBtn.clicked.connect(self._on_train_btn_click)
        self.ui.TestBtn.clicked.connect(self._on_test_btn_click)
        self.ui.Dataset.currentIndexChanged.connect(self._on_dataset_selected)
        self.ui.LoadCustomTest.clicked.connect(self._on_custom_test_btn_pushed)
        self.ui.UseDefaultTest.clicked.connect(self._on_use_default_test_data_btn_pushed)

    @pyqtSlot()
    def _on_train_btn_click(self):
        self.ui.TrainBtn.setEnabled(False)
        self.ui.TestBtn.setEnabled(False)
        self.ui.Dataset.setEnabled(False)
        self.train_btn_click_signal.emit()

    @pyqtSlot()
    def _on_test_btn_click(self):
        self.ui.TrainBtn.setEnabled(False)
        self.ui.TestBtn.setEnabled(False)
        self.ui.Dataset.setEnabled(False)
        QCoreApplication.processEvents()
        self.test_btn_click_signal.emit()

    @pyqtSlot()
    def _on_use_default_test_data_btn_pushed(self):
        self.use_default_test_signal.emit()

    @pyqtSlot(int)
    def _on_dataset_selected(self, i):
        self.ui.TrainBtn.setEnabled(False)
        self.ui.TestBtn.setEnabled(False)
        self.ui.Dataset.setEnabled(False)
        dataset_name = self.ui.Dataset.currentText()
        self.dataset_select_signal.emit(dataset_name)

    @pyqtSlot()
    def _on_custom_test_btn_pushed(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        if dlg.exec_():
            dirname = dlg.selectedFiles()[0]
            self.load_custom_test_signal.emit(dirname)
            self.ui.UseDefaultTest.setEnabled(True)

    def enable_ui(self):
        self.ui.TrainBtn.setEnabled(True)
        self.ui.TestBtn.setEnabled(True)
        self.ui.Dataset.setEnabled(True)

    def show_prediction(self, result, image_list):
        for i in reversed(range(self.ui.formLayout.count())):
            self.ui.formLayout.itemAt(i).widget().setParent(None)
        for i, val in enumerate(result):
            label = QLabel()
            pred_class = val['class_ids'][0]
            pred_confidence = val['probabilities'][pred_class] * 100
            height, width, channels = image_list[i].shape
            bytesPerLine = channels * width
            if channels == 1:
                qImg = QImage(image_list[i], width, height, bytesPerLine, QImage.Format_Grayscale8)
            else:
                qImg = QImage(image_list[i], width, height, bytesPerLine, QImage.Format_RGB888)
            qImg = qImg.scaled(height * 2, width * 2, Qt.KeepAspectRatio)
            pixmap = QPixmap.fromImage(qImg)
            label.setPixmap(pixmap)
            text_label = QLabel()
            text_label.setText('Prediction Class: ' + str(pred_class) + '\n' + 'Confidence: ' + str(pred_confidence))
            self.ui.formLayout.addRow(label, text_label)

    def collect_config(self):
        config = {
            'learning_rate': self.ui.LearningRate.toPlainText(),
            'train_steps': self.ui.TrainStep.toPlainText(),
            'adanet_lambda': self.ui.Lambda.toPlainText(),
            'learn_mixture_weights': self.ui.LearnMixtureWeight.currentText(),
            'adanet_iteration': self.ui.AdanetIteration.toPlainText(),
            'generator': self.ui.Generator.currentText(),
        }
        return config
