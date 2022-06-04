import cv2
import numpy as np

img = cv2.imread('prueba.jpg', cv2.IMREAD_COLOR)
cv2.imshow('Deteccion de colores', img)

# Convert to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range of blue color in HSV
lower_bound = np.array([50,20,20])
upper_bound = np.array([70,255,255])

mask = cv2.inRange(hsv, lower_bound, upper_bound)

kernel = np.ones((7,7), np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

cv2.imshow('Mask', mask)

segmented_img = cv2.bitwise_and(img, img, mask=mask)

contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = cv2.drawContours(img, contours, -1, (0,0,255), 3)

cv2.imshow('Output', output)

cv2.waitKey(0)
cv2.destroyAllWindows()