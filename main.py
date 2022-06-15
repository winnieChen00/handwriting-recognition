#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image


# In[2]:


def video_loop():
    cap = cv2.VideoCapture(0)                     # 啟用攝影鏡頭
    if not cap.isOpened():
        exit()
    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.resize(img,(540,300))          # 改變影像尺寸，加快處理效率
        x, y, w, h = 400, 200, 60, 60            # 定義擷取數字的區域位置和大小
        img_num = img.copy()                     # 複製一個影像辨識
        img_num = img_num[y:y+h, x:x+w]          # 擷取辨識的區域

        img_num = cv2.cvtColor(img_num, cv2.COLOR_BGR2GRAY)    # 顏色轉成灰階
        ret, img_num = cv2.threshold(img_num, 127, 255, cv2.THRESH_BINARY_INV)# 轉成黑底白字
        img_num = cv2.resize(img_num,(28,28))   # 縮小成 28x28，和訓練模型對照
        img_num = img_num.astype(np.float32)    # 轉換格式
        img_num = img_num.reshape(-1,)          
        img_num = img_num.reshape(1,-1)
        img_num = img_num/255
        img_pre = knn.predict(img_num)          # 進行辨識
        num = str(int(img_pre[1][0][0]))        # 取得辨識結果
 
        cv2.putText(img, num, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA) # 印出辨識結果
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)  # 標記辨識的區域
        
        
        cv2.putText(img, 'press q to stop', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if cv2.waitKey(50) == ord('q'): # 按下 q 鍵停止
            break
            
        cv2.imshow('Starting...', img)
    cap.release()


# In[3]:


def predict(filePath):
    img = cv2.imread(filePath)
    img_num = img.copy()                     # 複製一個影像作為辨識使用
    img_num = cv2.cvtColor(img_num, cv2.COLOR_BGR2GRAY)    # 顏色轉成灰階

    
    ret, img_num = cv2.threshold(img_num, 127, 255, cv2.THRESH_BINARY_INV)# 轉成黑底白字

    img_num = cv2.resize(img_num,(28,28))   # 縮小成 28x28，和訓練模型對照

    img_num = img_num.astype(np.float32)    # 轉換格式
    img_num = img_num.reshape(-1,)          
    img_num = img_num.reshape(1,-1)
    img_num = img_num/255
    
    img_pre = knn.predict(img_num)          # 進行辨識
    num = str(int(img_pre[1][0][0]))        # 取得辨識結果
    result1['text']='辨識結果：'+str(num)    # 顯示結果


# In[4]:


#設定GUI視窗
win = tk.Tk()
win.title('手寫數字辨識') 
win.geometry('260x380')

knn = cv2.ml.KNearest_load('./mnist_knn.xml')   # 載入先前訓練好的模型

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img

def oas():
    result1['text']='' # 前面辨識的結果先刪除
    global sfname      # 檔案位置
    sfname = filedialog.askopenfilename(title='選擇',
                                        filetypes=[
                                            ('All Files','*'),
                                            ("jpeg files","*.jpg"),
                                            ("png files","*.png"),
                                            ("gif files","*.gif")]) # 選擇檔案

    im = cv_imread(sfname) # 讀取檔案
    cv2image = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    image.imgtk = imgtk
    image.configure(image=imgtk)
    
def identify():
    predict(sfname)

B1 = tk.Button(win, text="開啟檔案",command = oas)
B1.grid(row=2, padx=100, pady=2)

label = tk.Label(win, text='或')
label.grid(row=1, column=0 ,padx=10, pady=2)

videoFrame = tk.Frame(win)
videoFrame.grid(row=3)

image = tk.Label(videoFrame)
image.grid(row=4, pady=2)

result1 = tk.Label(win, text='')
result1.grid(row=6, column=0 ,padx=10, pady=2)

B2 = tk.Button(win, text="辨識",command = identify)
B2.grid(row=5, column=0 ,padx=100, pady=2)

B3 = tk.Button(win, text="開啟攝影機",command = video_loop)
B3.grid(row=0, padx=100, pady=2)

win.mainloop()

