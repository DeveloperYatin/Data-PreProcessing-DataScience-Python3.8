from PyQt5.QtWidgets import QApplication,QMessageBox
from PyQt5 import uic
import joblib
import numpy as np

def predict():
    if(dlg.Experience.text() == ""):
        Show_Message("Warning!!","Enter some value first!!")
    else:    
        a = float(dlg.Experience.text())
        pred_b = [a]
        pred_b_arr = np.array(pred_b)
        pred_b_arr = pred_b_arr.reshape(1, -1)
        linear_reg = open("linear_regression_model.pkl","rb")
        ml_model = joblib.load(linear_reg)
        model_prediction = ml_model.predict(pred_b_arr)
        dlg.Salary.setText(str(round(float(model_prediction), 2)))
		
def Show_Message(title="title",message="message"):
	QMessageBox.information(None,title,message)
	

	
if __name__ == '__main__':
	app = QApplication([])
	dlg = uic.loadUi("linear_regression_ui.ui")
	dlg.Experience.setFocus()
	dlg.GetSalary.clicked.connect(predict)
	dlg.Experience.returnPressed.connect(predict)
	dlg.Salary.setReadOnly(True)
	dlg.show()
	app.exec()
