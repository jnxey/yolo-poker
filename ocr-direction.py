import easyocr

reader = easyocr.Reader(['en'])  # 英文 + 中文
result = reader.readtext('t30.png')
for bbox, text, prob in result:
    print(text, prob)
