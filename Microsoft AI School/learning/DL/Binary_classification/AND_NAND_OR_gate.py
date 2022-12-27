# 단층 퍼셉트론으로 AND NAND OR 게이트 구현 가능

# AND 게이트 구현
# AND 게이트를 만족하는 두 개의 가중치와 편향에는 무엇이 있는가?
# w1 w2 b
# 수많은 실험 결과 구해진 값 [0.5, 0.5, -0.7]

def AND_gate(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.7

    result = x1 * w1 + x2 * w2 + b
    if result <= 0:
        return 0
    else:
        return 1


print(AND_gate(1, 1))

# NAND 게이트

def NAND_gate(x1, x2):
    w1, w2, b = -0.5, -0.5, 0.7

    result = x1 * w1 + x2 * w2 + b
    if result <= 0:
        return 0
    else:
        return 1


print(NAND_gate(1, 1))

# OR 게이트

def OR_gate(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.4

    result = x1 * w1 + x2 * w2 + b
    if result <= 0:
        return 0
    else:
        return 1


print(OR_gate(1, 0))


