"""$ python cochrans_q_test.py
Cochran's Q test
q=7.529411764705882
significance level = 0.05
p_value=0.023174427241061245
"""
# ref: https://rasbt.github.io/mlxtend/user_guide/evaluate/cochrans_q/#example-1-cochrans-q-test
import numpy as np
from mlxtend.evaluate import cochrans_q

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
y_model_3 = np.array(
    [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    + [0] * 76
    + [1, 1]
)

print("Cochran's Q test")
q, p_value = cochrans_q(y_true, y_model_1, y_model_2, y_model_3)
print(f"{q=}")
print("significance level = 0.05")
print(f"{p_value=}")
