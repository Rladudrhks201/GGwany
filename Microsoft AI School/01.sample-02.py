import socket

in_addr = socket.gethostbyname(socket.gethostname())

print(in_addr)

# 현재 접속중인 IP를 출력하는 코드