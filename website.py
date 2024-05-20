import os
import cv2
import numpy as np
import dlib
import streamlit as st

# Define the directory where the model is located
model_directory = ""  # Replace with the actual directory path

# Create the full file path using os.path.join
model_path = os.path.join(model_directory, "shape_predictor_68_face_landmarks.dat")

# Load the model
predictor = dlib.shape_predictor(model_path)
detector = dlib.get_frontal_face_detector()

st.title("Face Landmarks Streamlit App")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

st.subheader("Adjust Lip Color:")
red_value = st.slider("Red", 0, 255, 255)
green_value = st.slider("Green", 0, 255, 0)
blue_value = st.slider("Blue", 0, 255, 0)

if uploaded_image is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    finalimg = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.image(img, caption="Original Image", use_column_width=True)
    st.image(imgGray, caption="Gray Image", use_column_width=True)

    faces = detector(imgGray)
    landmarkspoints = []

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        st.image(img, caption="Face Detected", use_column_width=True)

        landmarks = predictor(imgGray, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarkspoints.append([x, y])
            cv2.circle(img, (x, y), 3, (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(n), (x + 1, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)

    st.image(img, caption="Face Landmark", use_column_width=True)

    landmarkspoints = np.array(landmarkspoints)
    lipmask = np.zeros_like(img)
    lipimg = cv2.fillPoly(lipmask, [landmarkspoints[48:60]], (255, 255, 255))
    lipimgcolor = np.zeros_like(lipimg)
    b = blue_value
    g = green_value
    r = red_value
    lipimgcolor[:] = b, g, r
    lipimgcolor = cv2.bitwise_and(lipimg, lipimgcolor)

    kernel = np.ones((5, 5), np.uint8)
    lipimgcolor = cv2.erode(lipimgcolor, kernel, iterations=1)
    lipimgcolor = cv2.GaussianBlur(lipimgcolor, (7, 7), 10)

    st.image(lipimgcolor, caption="Final Image with Lips after Erosion", use_column_width=True)

    finalimg = cv2.addWeighted(finalimg, 1, lipimgcolor, 0.6, 0)
    st.image(finalimg, caption="Final Image", use_column_width=True)
