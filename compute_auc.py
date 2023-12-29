import numpy as np
from sklearn import metrics

# K (number of shots)
x = np.array([1., 30., 50., 100., 300.])
x_log = np.log(x) / np.log(300)
# Average Recall scores
y = np.array([0.0, 41.86, 43.12, 44.46, 46.02])
y *= 0.01
auc = metrics.auc(x_log, y)
print('AUC score:', auc)
print(360081 + 68567 + 2364400 + 276928 + 32529 + 318443 + 168754 +
139148 + 40706 + 1213397 + 108976 + 44085 + 190695 + 85658 +
410028 + 51927 + 1151936 + 100046 + 26457 + 200875 + 154186)