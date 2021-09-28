import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMAResults
import matplotlib.pyplot as plt

model_fit = ARIMAResults.load('ARIMA_Model.pkl')
model_fit.plot_predict(dynamic=False)
plt.show()
