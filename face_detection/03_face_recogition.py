import cv2
import os 
import numpy as np
import time
#from picamera2 import Picamera2

#Parameters
id = 0
font = cv2.FONT_HERSHEY_COMPLEX
height=1
boxColor=(0,0,255)      #BGR- GREEN
nameColor=(255,255,255) #BGR- WHITE
confColor=(255,255,0)   #BGR- TEAL

BASE_DIR = os.path.dirname(__file__)
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_path = os.path.join(BASE_DIR, 'trainer', 'trainer.yml')
if not os.path.exists(model_path):
    print(f"\n[ERROR] Model file not found: {model_path}")
    raise SystemExit(1)
recognizer.read(model_path)
# names related to id
names = ['None', 'Vinh']
"""
# Create an instance of the PiCamera2 object
cam = Picamera2()
## Initialize and start realtime video capture
# Set the resolution of the camera preview
cam.preview_configuration.main.size = (640, 360)
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.controls.FrameRate=30
cam.preview_configuration.align()
cam.configure("preview")
cam.start()
"""

pTime = 0
cam = cv2.VideoCapture('Vinh_Test.mp4') #Test thu tren may tinh
while True:
    # Capture a frame from the camera
    #frame=cam.capture_array()

    ret, frame = cam.read()  #Test thu tren may tinh
    if not ret:
        print("Failed to grab frame")
        break

    #Convert frame from BGR to grayscale
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Create a DS faces- array with 4 elements- x,y coordinates top-left corner), width and height
    faces = face_detector.detectMultiScale(
            frameGray,      # The grayscale frame to detect
            scaleFactor=1.1,# how much the image size is reduced at each image scale-10% reduction
            minNeighbors=5, # how many neighbors each candidate rectangle should have to retain it
            minSize=(150, 150)# Minimum possible object size. Objects smaller than this size are ignored.
            )
    for(x,y,w,h) in faces:
        namepos=(x+5,y-5) #shift right and up/outside the bounding box from top
        confpos=(x+5,y+h-5) #shift right and up/intside the bounding box from bottom
        #create a bounding box across the detected face
        cv2.rectangle(frame, (x,y), (x+w,y+h), boxColor, 3) #5 parameters - frame, topleftcoords,bottomrightcooords,boxcolor,thickness

        try:
            label_id, distance = recognizer.predict(frameGray[y:y+h,x:x+w])
        except Exception as e:
            print(f"[WARN] recognizer.predict failed: {e}")
            continue

        # If confidence is less than 70, it is considered a match
        conf_score = 100 - min(distance, 100)
        if distance < 70 and 0 <= label_id < len(names):
            label = names[label_id]
            confidence = f"{conf_score:.0f}%"
        else:
            label = "unknown"
            confidence = f"{100 - conf_score:.0f}%"

        #Display name and confidence of person who's face is recognized
        cv2.putText(frame, str(label), namepos, font, height, nameColor, 2)
        cv2.putText(frame, str(confidence), confpos, font, height, confColor, 1)

    #Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (5, 25), font, 1, (0, 255, 0), 2)
    cv2.imshow('Raspi Face Recognizer',frame)

    # Wait for 30 milliseconds for a key event (extract sigfigs) and exit if 'ESC' or 'q' is pressed
    key = cv2.waitKey(100) & 0xff
    #Checking keycode
    if key == 27:  # ESCAPE key
        break
    elif key == 113:  # q key
        break

print("\n [INFO] Exiting Program and cleaning up stuff")
cam.release()
cv2.destroyAllWindows()
