import numpy as np
import pickle
import cv2
import time
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import tkinter.scrolledtext as tkst

min_confidence = 0.5
width = 800
height = 0
show_ratio = 1.0
file_name = 'C:/Users/ITWILL/Desktop/custom/apple_1_084.jpg'
weight_name = 'C:/Users/ITWILL/Desktop/custom/custom-train-yolo_final.weights'
cfg_name = 'C:/Users/ITWILL/Desktop/custom/custom-train-yolo.cfg'
classes_name = 'C:/Users/ITWILL/Desktop/custom/classes.names'
title_name = 'Apple vs Diseased apple'
classes = []


def detectAndDisplay(image):
    net = cv2.dnn.readNet(weight_name, cfg_name)  # weights, cfg 파일을 불러와서 yolo 네트워크와 연결
    with open(classes_name, "r") as f:
        classes = [line.strip() for line in f.readlines()]  # classes.names을 불러와 classes 배열에 넣기
    # color_lists = np.random.uniform(0, 255, size=(len(classes), 3))  # 클래스 별 색 구분.
    color_lists = [(0, 255, 0), (0,0,255)]
    layer_names = net.getLayerNames()  # 네트워크의 모든 레이어 이름을 가져와서 layer_names에 넣기
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # 레이어 중 출력 레이어 인덱스 가져오기
    h, w = image.shape[:2]
    height = int(h * width / w)  # width height 맞추기 위해 설정.
    img = cv2.resize(image, (width, height))

    #  물체 감지
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)  # 이미지를 blob 객체로 처리

    net.setInput(blob)  # blob 객체에 setInput 함수 적용
    outs = net.forward(output_layers)  # output_layers를 네트워크 순방향으로 추론

    confidences = []  # 0에서 1 까지 사물 인식에 대한 신뢰도를 넣는 배열
    names = []  # 인식한 사물 클래스 이름을 넣는 배열.
    boxes = []  # 사물을 인식해서 그릴 상자에 대한 배열.
    colors = []  # 상자 및 글의 색에 대한 배열.

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)  # scores 중에서 최대값을 찾아 class_id에 넣음.
            confidence = scores[class_id]  # scores 중에서 class_id에 해당하는 값을 confidence에 넣음
            if confidence > min_confidence:  # 정확도 0.5(min_confidence)가 넘는다면 사물이 인식되었다고 판단
                # 객체 탐지
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                names.append(classes[class_id])
                colors.append(color_lists[class_id])

    # 노이즈 제거(같은 사물에 대해서 박스가 여러 개인 것을 제거하는 NonMaximum Suppresion 작업)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN  # Font 적용

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = '{} {:,.2%}'.format(names[i], confidences[i])  # 클래스 이름, 정확도를 label에 저장
            color = colors[i]  # 박스, 레이블 색 넣기
            print(i, label, x, y, w, h)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # 박스 그리기
            cv2.putText(img, label, (x+10, y + 20), font, 1, color, 2)  # yolo가 인식한 사물 클래스 출력

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 이미지 RGB로 바꿔서 출력
    image = Image.fromarray(image)  # numpy배열로 되어있는 image를 image 객체로 변환
    imgtk = ImageTk.PhotoImage(image=image)  # tkinter와 호환되는 객체로 변환
    detection_image.config(image=imgtk)  #
    detection_image.image = imgtk

def selectFile():
    file_name = filedialog.askopenfilename(initialdir = "./", title = "Select image file",
                                           filetypes = (("jpg files", "*.jpg"), ("all files", "*.*")))
    read_image = cv2.imread(file_name)
    file_path['text'] = file_name
    detectAndDisplay(read_image)

main = Tk()
main.title(title_name)
main.geometry()

read_image = cv2.imread(file_name)
image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
imgtk = ImageTk.PhotoImage(image=image)

label = Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0, column=0, columnspan=4)

file_title = Label(main, text="Image")
file_title.grid(row=1, column=0, columnspan=1)
file_path = Label(main, text=file_name)
file_path.grid(row=4, column=1, columnspan=2)
Button(main, text='Select', height=1, command=lambda:selectFile()).grid(row=1, column=3, columnspan=1, sticky=(N,S,W,E))


detection_image = Label(main, image=imgtk)
detection_image.grid(row=2, column=0, columnspan=4)

detectAndDisplay(read_image)

main.mainloop()