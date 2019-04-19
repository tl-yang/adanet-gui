import sys
from controller import Controller
from PyQt5.QtWidgets import QApplication
from model.backend_model import BackendModel

from main_window import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    model = BackendModel()
    controller = Controller(window, model)
    window.show()
    app.exec_()
