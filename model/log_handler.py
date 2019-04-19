import logging

from PyQt5 import QtCore


class QtHandler(logging.Handler):
    def __init__(self):
        self.event = QtLogEvent()
        logging.Handler.__init__(self)

    def emit(self, record):
        record = self.format(record)
        if record:
            self.event.receive_messsage(record)


class QtLogEvent(QtCore.QObject):
    message_received = QtCore.pyqtSignal(str)

    def receive_messsage(self, record):
        self.message_received.emit(record)
