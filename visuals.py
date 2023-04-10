import cv2
import numpy as np

def draw_indicator(img_cp, img, window_list, w=100, h=80, x=30, y=30):
    for i, bbox in enumerate(window_list):
        #a roundabout way of displaying BRAKE if there are any windows currently being drawn
        cv2.putText(img_cp, 'BRAKE', (400,37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        break
