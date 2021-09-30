import pickle
import numpy as np
model = pickle.load(open("lasso_regression.sav", mode="rb"))
arr = np.array(500)
print(model.predict(arr.reshape(-1,1)))
