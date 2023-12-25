import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier 
import numpy as np
import math
from keras.models import load_model
import time
import speak

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
folder = "Data"
counter = 0
acc=0
global class_name
global confidence_score
# lables = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","1","2","3","4","5","6","7","8","9"," "]
#new model
np.set_printoptions(suppress=True)
# Load the model
model = load_model("Model/keras_model.h5", compile=False)
# Load the labels
class_names = open("Model/labels.txt", "r").readlines()
while True:
    success, img = cap.read()
    imgoutput=img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgWhite =np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]
        #imgcrop=cv2.resize(imgCrop,(300,300))
        imgCropShape = imgCrop.shape
        aspectRatio= h/w
        if aspectRatio >1:
            k= imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((300-wCal) /2 )
            imgWhite[0:imgResizeShape[0],wGap:wCal+wGap]=imgResize
            imgWhite=cv2.resize(imgWhite, (224, 224), interpolation=cv2.INTER_AREA)
            imgWhite = np.asarray(imgWhite, dtype=np.float32).reshape(1, 224, 224, 3)
            imgWhite = (imgWhite / 127.5) - 1
            prediction = model.predict(imgWhite)
            index = np.argmax(prediction)
            class_name = class_names[index]
            class_name=(class_name[2:])
            confidence_score = prediction[0][index]
            confidence_score=confidence_score*100
            if confidence_score>80:
                print("Class:", class_name, end="")
                print("Confidence Score:", str(np.round(confidence_score))[:-2], "%")
            else:
                pass           
        if aspectRatio <1:
            k= imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal) /2 )
            imgWhite[hGap:hCal+hGap, :]=imgResize
            imgWhite=cv2.resize(imgWhite, (224, 224), interpolation=cv2.INTER_AREA)
            imgWhite = np.asarray(imgWhite, dtype=np.float32).reshape(1, 224, 224, 3)
            imgWhite = (imgWhite / 127.5) - 1
            prediction = model.predict(imgWhite)
            index = np.argmax(prediction)
            class_name = class_names[index]
            class_name=class_name[2:]
            confidence_score = prediction[0][index]
            confidence_score=confidence_score*100
            if confidence_score>80:
                print("Class:", class_name)
                print("Confidence Score:", str(np.round(confidence_score))[:-2], "%")
            else:
                pass     
        #cv2.imshow("ImageCrop", imgCrop)
        #cv2.imshow("ImageWhite", imgWhite)
        if confidence_score>95:
            confidence_score=round(confidence_score,0) 
            cv2.putText(imgoutput,"Output: -"+' '.join(class_name)+' '.join(str(np.round(confidence_score))[:-2])+'%', (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)  
            cv2.imshow("Image", imgoutput)
            cv2.waitKey(2)
            speak.speak(f"{class_name} with {confidence_score} %")
        else:
            pass
    cv2.imshow("Image", imgoutput)
    cv2.waitKey(2)
