import qrcode

# qr_data = 'www.naver.com'
with open('C:/Users/user/Documents/github/site_list.txt','rt',encoding='UTF8') as f:
    link1 = f.readlines()
print(link1)
    
for i in link1:
    line = i.strip()
    qr_image = qrcode.make(line)

    qr_image.save( line +'.png')


