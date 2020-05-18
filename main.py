import sys

from PyQt5 import QtWidgets

from amt.GUI import Ui_MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = Ui_MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
