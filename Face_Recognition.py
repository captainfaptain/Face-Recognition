import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to images and loading images
path = r"D:\Project\Face_Recognition\images"
images = []
personNames = []
myList = os.listdir(path)
print(myList)

for cu_img in myList:
    current_Img = cv2.imread(os.path.join(path, cu_img))
    if current_Img is None:
        print(f"Warning: {cu_img} could not be loaded.")
        continue
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
print(personNames)

# Function to encode face images
def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:  # Check if any faces were found
            encodeList.append(encodes[0])
        else:
            print("Warning: No face encodings found in an image.")
    return encodeList

encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

# Initialize webcam with different index if needed
cap = cv2.VideoCapture(0)  # Try changing this to 1, 2, or -1
if not cap.isOpened():
    print("Error: Could not open webcam. Trying different index.")
    cap = cv2.VideoCapture(1)  # Try index 1 if available
    if not cap.isOpened():
        print("Error: Could not open webcam with index 1. Ensure the webcam is connected and working.")
        exit()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Warning: Failed to capture frame.")
        continue

    # Resize the frame
    faces = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            print(name)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Break the loop on 'Enter' key press
    if cv2.waitKey(1) == 13:  # Enter key
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
