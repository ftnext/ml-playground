# ref: https://rasbt.github.io/mlxtend/user_guide/evaluate/cochrans_q/#example-1-cochrans-q-test
import numpy as np
from mlxtend.evaluate import cochrans_q, mcnemar, mcnemar_table

y_true = np.array([0] * 100)
y_model_1 = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    + [0] * 76
    + [0, 0]
)
y_model_2 = np.array(
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    + [0] * 76
    + [0, 0]
)

assert cochrans_q(y_true, y_model_1, y_model_2) == mcnemar(
    mcnemar_table(y_true, y_model_1, y_model_2), corrected=False
)
