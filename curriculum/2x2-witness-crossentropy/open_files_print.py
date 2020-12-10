import numpy as np

array_100 = np.load("curriculum_budget_100.npy", allow_pickle=True)
array_200 = np.load("curriculum_budget_200.npy", allow_pickle=True)

print("array_100 =", array_100)
print("")
for l in array_100:
    print("len(l) =", len(l))
print("")
print("array_200 =", array_200)
for l in array_200:
    print("len(l) =", len(l))
print("")