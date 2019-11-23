# -*- coding:utf-8 -*-
import re
import requests


def dowmloadPic(html, keyword,n):
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)
    i = 1
    print('找到关键词:' + keyword + '的图片，现在开始下载图片...')
    for each in pic_url:
        print('正在下载第' +str(n)+"页第"+ str(i) + '张图片，图片地址:' + str(each))
        try:
            pic = requests.get(each, timeout=5)
        except requests.exceptions.ConnectionError:
            print('【错误】当前图片无法下载')
            continue

        dir = './img/' + keyword + '_'+str(n)+"_" + str(i) + '.jpg'
        fp = open(dir, 'wb')
        fp.write(pic.content)
        fp.close()
        i += 1


if __name__ == '__main__':
    n=1
    word = input("Input key word: ")
    while(n<50):
        url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + '&ct=201326592&v=flip&pn='+str(n*10)
        result = requests.get(url)
        try:
            dowmloadPic(result.text, word,n)
        except requests.exceptions.ReadTimeout:
            print("get html faild")
        n+=1

