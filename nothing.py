import numpy as np
import torch
import random


b = []
k = 0
for j in range(3):
    a = []
    for i in range(2):
        k += 1
        a.append(k)
    b.append(a)

print(a)
print("**********")
print(b)
print(len(b))
print("**********")

m = 0
for j in range(1, len(b)):
    m += 1
    print(m)

