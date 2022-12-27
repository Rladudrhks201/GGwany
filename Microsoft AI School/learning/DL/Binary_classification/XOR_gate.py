# 다중 퍼셉트론을 활용한 XOR gate 구현
import numpy as np


def and_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    temp = np.sum(w * x) + b
    if temp <=0:
        return 0
    else:
        return 1

def nand_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    temp = np.sum(w * x) + b
    if temp <=0:
        return 0
    else:
        return 1

def or_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.4

    temp = np.sum(w * x) + b
    if temp <= 0:
        return 0
    else:
        return 1

def xor_gate(x1, x2):
    s1 = nand_gate(x1, x2)
    s2 = or_gate(x1, x2)
    y = and_gate(s1, s2)

    return y