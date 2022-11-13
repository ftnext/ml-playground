"""$ python create_contingency_table.py
a (model1 & 2 right): 4
b (model1 right & model2 wrong): 2
c (model1 wrong & model2 right): 1
d (model1 & 2 wrong): 3

Contingency table:
[[4 2]
 [1 3]]
"""
# ref: https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_table/#example-2-2x2-contingency-table
import numpy as np
from mlxtend.evaluate import mcnemar_table

y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
y_model1 = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0])
y_model2 = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0])

# y_model1: 正, 誤, 正, 正, 正, 正, 正, 誤, 誤, 誤
# y_model2: 正, 正, 誤, 誤, 正, 正, 正, 誤, 誤, 誤
a = filter(
    lambda t: t[1] == t[0] and t[2] == t[0], zip(y_true, y_model1, y_model2)
)
print("a (model1 & 2 right):", len(list(a)))
b = filter(
    lambda t: t[1] == t[0] and t[2] != t[0], zip(y_true, y_model1, y_model2)
)
print("b (model1 right & model2 wrong):", len(list(b)))
c = filter(
    lambda t: t[1] != t[0] and t[2] == t[0], zip(y_true, y_model1, y_model2)
)
print("c (model1 wrong & model2 right):", len(list(c)))
d = filter(
    lambda t: t[1] != t[0] and t[2] != t[0], zip(y_true, y_model1, y_model2)
)
print("d (model1 & 2 wrong):", len(list(d)))
print()

tb = mcnemar_table(y_true, y_model1, y_model2)
print("Contingency table:")
print(tb)
