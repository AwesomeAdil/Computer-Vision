import cv2 as cv
import time 
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math
import osascript
import methods as mm
import screen_brightness_control as sbc
from base64 import b64decode
import sys
import os
import pyautogui as pg
import keyboard as kb
cap = cv.VideoCapture(0)
ptime = ctime = 0

# possible sounds are neutral, sound, and brightness
state = "brightness"


while cap.isOpened():
    ret, img = cap.read()
    if not ctime:
        h, w, c = img.shape
    if ret:
        drawn_img, results, count, finger_status, hand = mm.fingerCounter(img, mm.hands, True)
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        dist = -1
        cv.putText(drawn_img,f'FPS: {int(fps)}', (50, 70), mm.font, mm.fontScale,mm.fontColor,mm.thickness,mm.lineType)
        # Labeling the landmarks also calculates if thumbs is highest y coordinate
        if results.multi_hand_landmarks:
            for handLMS in results.multi_hand_landmarks:
                sx = int(handLMS.landmark[4].x * w)
                sy = int(handLMS.landmark[4].y * h)
                ex = int(handLMS.landmark[8].x * w)
                ey = int(handLMS.landmark[8].y * h)
                drawn_img = cv.line(drawn_img, (sx, sy), (ex, ey), (0,255,255), 2)
                dist = math.dist((sx, sy), (ex, ey))
        #cv.putText(drawn_img,"DIST APART "+ str(dist), (0,h//2), font, 5,(255,255,255),thickness,lineType)
        if state == "sound":
            if dist!=-1: 
                if dist <= 50:
                    target_volume = 0
                elif dist >= 500:
                    target_volume = 100
                else:
                    target_volume = int(((dist - 50)/450) * 100)
                osascript.osascript("set volume output volume {}".format(target_volume))
                drawn_img = cv.line(drawn_img, (sx, sy), (ex, ey), (0, 255, 0), thickness=5)
            cv.imshow("Webcam", drawn_img)
        if state == "brightness":
            if dist!=-1: 
                if dist <= 50:
                    kb.press('f1')
                    print("moo")
                elif dist >= 200:
                    print("foo")
                    kb.press('f2')
                
                drawn_img = cv.line(drawn_img, (sx, sy), (ex, ey), (0, 255, 0), thickness=5)
            
    
    
    cv.imshow("Webcam", drawn_img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()
