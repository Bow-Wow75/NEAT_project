import cv2

cap = cv2.VideoCapture('dashcam.MOV')

#classifier taken from here for now: https://github.com/afzal442/Real-Time_Vehicle_Detection-as-Simple/blob/master/cars.xml
car_cascade = cv2.CascadeClassifier('cars.xml')

while True:
   ret, frames = cap.read()
   gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
   
   cars = car_cascade.detectMultiScale(gray, 1.1, 1)
   
   #make rectangles
   for(x,y,w,h) in cars:
      cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
      
   cv2.imshow('video', frames)
   
   if cv2.waitKey(33) == 27:
     break;
