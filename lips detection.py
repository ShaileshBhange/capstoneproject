import os
import cv2
import numpy as np
import dlib

# Define the directory where the model is located
model_directory = ""  # Replace with the actual directory path

# Create the full file path using os.path.join
model_path = os.path.join(model_directory, "shape_predictor_68_face_landmarks.dat")

# Load the model

img = cv2.imread('pranjal photo.png')
img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
finalimg = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Original Image:")
cv2.imshow('Original Image', img)
cv2.waitKey(0)

print("Grey Image:")
cv2.imshow('Grey Image', imgGray)
cv2.waitKey(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)
faces = detector(imgGray)
landmarkspoints = []

for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    print("Face detected:")
    cv2.imshow('Face Detected', img)
    cv2.waitKey(0)

    landmarks = predictor(imgGray, face)
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarkspoints.append([x, y])
        cv2.circle(img, (x, y), 3, (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(n), (x + 1, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)

    print("Face Landmark:")
    cv2.imshow('Face Landmark', img)
    cv2.waitKey(0)

landmarkspoints = np.array(landmarkspoints)
lipmask = np.zeros_like(img)
lipimg = cv2.fillPoly(lipmask, [landmarkspoints[48:60]], (255, 255, 255))
lipimgcolor = np.zeros_like(lipimg)
b = 0
g = 0
r = 255
lipimgcolor[:] = b, g, r
lipimgcolor = cv2.bitwise_and(lipimg, lipimgcolor)

kernel = np.ones((5, 5), np.uint8)
lipimgcolor = cv2.erode(lipimgcolor, kernel, iterations=1)
lipimgcolor = cv2.GaussianBlur(lipimgcolor, (7, 7), 10)

print("Final Image with Lips after Erosion:")
cv2.imshow('Final Image with Lips after Erosion', lipimgcolor)
cv2.waitKey(0)

finalimg = cv2.addWeighted(finalimg, 1, lipimgcolor, 0.6, 0)
print("Final Image:")
cv2.imshow('Final Image', finalimg)
cv2.waitKey(0)

# Wait for a key press and then close all OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()
