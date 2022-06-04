import cv2

video = cv2.VideoCapture(2)

while(True):
  #capture video frame
  ret, frame = video.read();

  if ret == True:
    cv2.imshow('Video Show', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break;

#after loop release the cap object
video.release();
cv2.destroyAllWindows()
