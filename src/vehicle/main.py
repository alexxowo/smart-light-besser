from sys import maxsize
import cv2
import numpy as np

vehicle_cascade = cv2.CascadeClassifier('cars.xml')

image = cv2.VideoCapture(0);

while True:
  ret, frame = image.read()

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  vehicles = vehicle_cascade.detectMultiScale(gray,
                      scaleFactor = 1.06,
                      minNeighbors = 10)

  print('Founded Vehicles: ', vehicles)

  for (x,y,w,h) in vehicles:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

  if ret == True:
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break;

cv2.destroyAllWindows()
