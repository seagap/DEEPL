import os
rootdir = './img'
testlen=0
imglist={}
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if os.path.isfile(path):
        imglist[path]=os.path.getsize(path)
#print(imglist.values())
dict1 = sorted(imglist.items(),key=lambda x:x[1])
size=0
for i in dict1:
    if os.path.isfile(i[0]):
        if(size==os.path.getsize(i[0])):
            size=os.path.getsize(i[0])    
            os.remove(i[0])
            print("delete:"+i[0])
        else:
            size=os.path.getsize(i[0])    
            print("save"+i[0]+"size:"+str(size))
          

