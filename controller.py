from PyQt5.QtCore import QObject, QThread, pyqtSignal

import re


class ReturnObject(QObject):
    def __init__(self, val):
        self.val = val


class ThreadTraining(QThread):
    finished = pyqtSignal(object)

    def __init__(self, process, callback, parent=None):
        super().__init__(parent)
        self.process = process
        self.finished.connect(callback)

    def run(self) -> None:
        result = self.process()
        self.finished.emit(ReturnObject(result))


class Controller(object):
    def __init__(self, view, model):
        self.model = model
        self.view = view
        self.t = None
        self._connect()

    def load_dataset(self, filename):
        try:
            self.model.load_dataset(filename)
        except RuntimeError as _:
            self.view.append_text_in_textedit('Load Dataset failed')
            # log error to view
        self.view.append_text_in_textedit('Dataset Loaded')

    def load_target_img(self, dirname):
        try:
            self.model.set_testing_data(dirname)
        except RuntimeError as e:
            self.view.append_text_in_textedit('Load Image Failed')
            # log error to view
        self.view.append_text_in_textedit('Custom test data in {} Loaded'.format(dirname))
        self.view.ui.UseDefaultTest.setEnabled(True)

    def use_default_test(self):
        try:
            self.model.use_default_test()
        except RuntimeError as e:
            self.view.append_text_in_textedit(str(e))
            # log error to view
        self.view.append_text_in_textedit('Using Default Test Set')

    def train(self):
        if not self.model.dataset_loaded:
            self.view.append_text_in_textedit('Dataset has not been loaded, please load dataset first')
            self.view.append_text_in_textedit('Training Failed')
            self.view.enable_ui()
        else:
            self.model.config = self.view.collect_config()
            self.model.config = {k: re.sub(r"\s+", "", v, flags=re.UNICODE) for (k, v) in self.model.config.items()}
            self.t = ThreadTraining(self.model.train, self.view.enable_ui)
            self.t.finished.connect(self._finished_training)
            self.t.start()

    def test(self):
        if not self.model.trained:
            self.view.append_text_in_textedit('Model has not been trained')
            self.view.append_text_in_textedit('Prediction Failed')
        else:
            result, image = self.model.predict()
            image = [img.reshape(img.shape + (1,)) if len(img.shape) != 3 else img for img in image]
            image = [img.data for img in image]
            self.view.show_prediction(result, image)
            self.view.append_text_in_textedit('Prediction Finished')
        self.view.enable_ui()
        self.view.ui.TabWidget.setCurrentIndex(2)

    def select_dataset(self, dataset_name):
        valid = self.model.load_dataset(dataset_name)
        if valid:
            self.view.append_text_in_textedit('Dataset downloaded and loaded')
        self.view.ui.TrainBtn.setEnabled(True)
        self.view.ui.Dataset.setEnabled(True)
        self.view.ui.LoadCustomTest.setEnabled(False)

    def _connect(self):
        self.view.train_btn_click_signal.connect(self.train)
        self.view.test_btn_click_signal.connect(self.test)
        self.view.dataset_select_signal.connect(self.select_dataset)
        self.view.load_custom_test_signal.connect(self.load_target_img)
        self.view.use_default_test_signal.connect(self.use_default_test)
        self.model.log_handler.event.message_received.connect(self.view.append_text_in_textedit)

    def _finished_training(self, result):
        result = result.val
        self.view.append_text_in_textedit('Training finished !')
        self.view.enable_ui()
        self.view.ui.LoadCustomTest.setEnabled(True)
        self.view.append_text_in_textedit('Start Evaluation: ')
        self.view.append_text_in_textedit('Final Loss: ' + str(result['average_loss']))
        self.view.append_text_in_textedit('Final Accuracy: ' + str(result['accuracy']))
