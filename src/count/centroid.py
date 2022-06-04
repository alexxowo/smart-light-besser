import cv2
import numpy as np

from tracker import EuclidianDistTracker

tracker = EuclidianDistTracker()

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    height, width = blur.shape
    print(height, width)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for contour in contours:
      (x, y, w, h) = cv2.boundingRect(contour)
      if cv2.contourArea(contour) < 300:
        continue
      detections.append([x, y, w, h])

    boxes_ids = tracker.update(detections)

    for box_id in boxes_ids:
      x,y,w,h,id = box_id
      cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Tracking", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break
