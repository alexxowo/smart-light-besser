from sys import maxsize
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

image = cv2.VideoCapture(0);

while True:
  ret, frame = image.read()

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = face_cascade.detectMultiScale(gray,
                      scaleFactor = 1.25,
                      minNeighbors = 5,
                      minSize = (80, 80),
                      maxSize = (400, 400))

  # print('Founded faces: ', faces.shape[0])

  for (x,y,w,h) in faces:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

  if ret == True:
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break;

cv2.destroyAllWindows()
