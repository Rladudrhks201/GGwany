import random as rd

random_number = rd.randint(1,100)

# print(random_number)

game_count = 1

while True:
    try:
        my_number = int(input("1~100 사이의 숫자를 입력하세요."))
    except:
        my_number = int(input('숫자를 입력해주세요.'))

    if my_number > random_number:
        print('Down')
    elif my_number < random_number:
        print('Up')
    else:
        print(f'축하합니다. {game_count} 만에 맞추셨습니다.')
        break
    game_count += 1

# 난수를 생성한 후 난수를 맞추는 게임을 생성하는 코드