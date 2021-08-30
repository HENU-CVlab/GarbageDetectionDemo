import sys
from PyQt5.QtWidgets import QApplication
from myWindow import myWindow
app = QApplication(sys.argv)
mainWindow = myWindow()
mainWindow.show()
sys.exit(app.exec())
