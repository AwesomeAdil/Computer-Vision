import cv2 as cv
import time 
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math
import osascript

# hands class
mp_hands = mp.solutions.hands

# setting up hand functions
hands = mp_hands.Hands(min_detection_confidence = 0.70)

# drawing class
mp_draw = mp.solutions.drawing_utils

# Text
font                   = cv.FONT_HERSHEY_SIMPLEX
fontScale              = 2
fontColor              = (255,255,0)
thickness              = 5
lineType               = 2


# Returns resulting image and results
def detectHandLandmarks(img, hands, draw=True):
    output_img = img.copy()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks and draw:
        for hand_landmark in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image = output_img, landmark_list = hand_landmark,
                                      connections = mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec = mp_draw.DrawingSpec(color=(0,0,255),
                                                                                  thickness=5, circle_radius=5),
                                      connection_drawing_spec=mp_draw.DrawingSpec(color = (255,255,255),
                                                                                     thickness = 2, circle_radius = 10))
    return (output_img, results)


def fingerCounter(img, hands, draw = True):
    finger_status = {
    "RIGHT THUMB": False,
    "RIGHT INDEX": False,
    "RIGHT MIDDLE": False,
    "RIGHT RING": False,
    "RIGHT PINKY": False,
    "LEFT THUMB": False,
    "LEFT INDEX": False,
    "LEFT MIDDLE": False,
    "LEFT RING": False,
    "LEFT PINKY": False}
    
    finger_tips = {"THUMB": 4,
                   "INDEX": 8,
                   "MIDDLE": 12,
                   "RING": 16,
                   "PINKY": 20}
 
    output_img = img.copy()
    rgbimg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = hands.process(rgbimg)
    hand = {"RIGHT": False, "LEFT": False}
    counts = {"RIGHT": 0, "LEFT": 0}
    if result.multi_handedness:
        for hand_index, handLmk in enumerate(result.multi_handedness):
            hand_label = handLmk.classification[0].label.upper()
            hand[hand_label] = True
            hand_landmarks = result.multi_hand_landmarks[hand_index]
            wrist = (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y)
            for key, val in finger_tips.items():
                finger = hand_label + " " + key
                tip = (hand_landmarks.landmark[val].x, hand_landmarks.landmark[val].y)
                if key == "THUMB":
                    knuckle = (hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y)
                    joint = (hand_landmarks.landmark[3].x, hand_landmarks.landmark[3].y)
                    finger_status[finger] = math.dist(tip, knuckle) > math.dist(joint, knuckle)
                else:
                    joint1 = (hand_landmarks.landmark[val - 1].x, hand_landmarks.landmark[val - 1].y)
                    joint2 = (hand_landmarks.landmark[val - 1].x, hand_landmarks.landmark[val - 1].y)
                    finger_status[finger] = math.dist(tip, wrist) > math.dist(joint, wrist)
#                 if math.dist(tip, wrist) > math.dist(joint, wrist) - delta:
#                     print("YAy", finger, finger_status[finger])
               
                if finger_status[finger]:
                    counts[hand_label] += 1


    if result.multi_hand_landmarks and draw:
        for hand_landmark in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image = output_img, landmark_list = hand_landmark,
                                      connections = mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec = mp_draw.DrawingSpec(color=(0,0,255),
                                                                                  thickness=5, circle_radius=5),
                                      connection_drawing_spec=mp_draw.DrawingSpec(color = (255,255,255),
                                                                                     thickness = 10, circle_radius = 10))
    return output_img, result, counts, finger_status, hand

