"""$ python mcnemar_test.py
McNemar test for Scenario B
[[9945   25]
 [  15   15]]
chi2=2.025
significance level = 0.05
p=0.15472892348537437

McNemar test for Scenario A
[[9959   11]
 [   1   29]]
chi2=None
significance level = 0.05
p=0.00634765625
"""
import numpy as np
from mlxtend.evaluate import mcnemar

# ref: https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/#example-2-mcnemars-test-for-scenario-b
table_scenario_b = np.array([[9945, 25], [15, 15]])
print("McNemar test for Scenario B")
print(table_scenario_b)
chi2, p = mcnemar(table_scenario_b)
print(f"{chi2=}")
print("significance level = 0.05")
print(f"{p=}")

print()

# ref: https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/#example-3-mcnemars-test-for-scenario-a
table_scenario_a = np.array([[9959, 11], [1, 29]])
print("McNemar test for Scenario A")
print(table_scenario_a)
chi2, p = mcnemar(table_scenario_a, exact=True)
print(f"{chi2=}")
print("significance level = 0.05")
print(f"{p=}")
