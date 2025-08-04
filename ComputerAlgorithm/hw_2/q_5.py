# 주사위 Dice(x) 를 구현. 1과 x 사이의 정수 값을 동일환 확률로 도출하는 함수. 예를 들어, Dice(x)는 1 이상의 99 이하의 정수를 1/99
# 의 동일한 확률로 반환하게 된다. 0과 1을 동일한 확률로 반환하는 함수 Bin()만 가지고 random api를 사용하지 않고, Bin() 함수와 기본
# 적인 사칙연산 및 비교만을 활용하여 Dice(x)를 구현하는 과정을 작성하시오. 
import random

# Bin 함수: 0과 1을 동일한 확률로 반환
def Bin():
    return random.randint(0, 1)

# Dice 함수: Bin() 함수를 사용하여 1과 x 사이의 정수를 동일한 확률로 반환
def Dice(x):
    # 1부터 x까지 동일한 확률로 선택할 수 있는 이진수 비트 생성
    num_bits = 1
    while (1 << num_bits) < x:  # num_bits는 x를 표현할 수 있는 최소 비트 수
        num_bits += 1

    while True:
        # num_bits만큼 이진수를 만들어서 무작위 숫자 생성
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | Bin()  # Bin() 결과를 왼쪽으로 이동하여 추가

        # 생성된 value가 x 이하일 경우 반환
        if value < x:
            return value + 1  # 결과값을 1부터 x 사이로 맞추기 위해 +1
