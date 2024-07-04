import numpy as np
from sklearn import metrics

# K (number of shots)
x = np.array([1.,10., 20., 30., 50., 100., 300.])
x_log = np.log(x) / np.log(300)
# Average Recall scores
y = np.array([0.0, 42.14, 43.46, 44.01, 44.56, 44.95, 45.01])
y *= 0.01
auc = metrics.auc(x_log, y)
print('AUC score:', auc)
