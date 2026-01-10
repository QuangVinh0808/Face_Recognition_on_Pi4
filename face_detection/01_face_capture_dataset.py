import cv2
import os
#from picamera2 import Picamera2

# Constants
COUNT_LIMIT = 150  #number of face samples to take
POS=(30,60)  #top-left
FONT=cv2.FONT_HERSHEY_COMPLEX #font type for text overlay
HEIGHT=1.5  #font_scale
TEXTCOLOR=(0,0,255)  #BGR- RED
BOXCOLOR=(255,0,255) #BGR- BLUE
WEIGHT=3  #font-thickness

# Ensure the Haar cascade is loaded from the script directory (avoids cwd issues)
BASE_DIR = os.path.dirname(__file__)
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
FACE_DETECTOR = cv2.CascadeClassifier(CASCADE_PATH)
if FACE_DETECTOR.empty():
    print(f"\n[ERROR] Could not load cascade classifier at {CASCADE_PATH}")
    raise SystemExit(1)

# For each person, enter one numeric face id
face_id = input('\n----Enter User-id and press <return>----')
print("\n [INFO] Initializing face capture. Look at the camera and wait!")

# Create an instance of the PiCamera2 object
"""
cam = Picamera2()
## Set the resolution of the camera preview
cam.preview_configuration.main.size = (640, 360)
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.controls.FrameRate=30
cam.preview_configuration.align()
cam.configure("preview")
cam.start()
"""
count=0
cam = cv2.VideoCapture(0) #Test thu tren may tinh
# Verify camera opened successfully
if not cam.isOpened():
    print("\n[ERROR] Could not open video device (camera). Check connection or try a different device index.")
    cam.release()
    cv2.destroyAllWindows()
    raise SystemExit(1)

while cam.isOpened():
    # Capture a frame from the camera
    #frame=cam.capture_array()

    ret, frame = cam.read()  #Test thu tren may tinh
    if not ret:
        print("Failed to grab frame")
        break
    cv2.putText(frame,'Count:'+str(int(count)),POS,FONT,HEIGHT,TEXTCOLOR,WEIGHT)

    # Convert frame from BGR to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR.detectMultiScale( # detectMultiScale has 4 parameters
            frame_gray,      # The grayscale frame to detect
            scaleFactor=1.1,# scale
            minNeighbors=5, # neighbors each candidate rectangle should have to retain it
            minSize=(20, 20)# Minimum possible object size
    )
    for (x,y,w,h) in faces:
        # Create a bounding box across the detected face
        cv2.rectangle(frame, (x,y), (x+w,y+h), BOXCOLOR, 3) # 5 parameters - frame, topleftcoords,bottomrightcooords,boxcolor,thickness
        count += 1 # increment count

        if not os.path.exists("face-detection\dataset"):
            os.makedirs("face-detection\dataset")
        # Save the captured bounded-grayscaleimage into the datasets folder only if the same file doesn't exist
        file_path = os.path.join("face-detection\dataset", f"User.{face_id}.{count}.jpg")
        if os.path.exists(file_path):
            # Ensure the old_dataset folder exists before moving
            old_dir = os.path.join(BASE_DIR, 'old_dataset')
            if not os.path.exists(old_dir):
                os.makedirs(old_dir)
            # Move the existing file to the "old_dataset" folder (preserve filename only)
            old_file_path = os.path.join(old_dir, os.path.basename(file_path))
            try:
                os.rename(file_path, old_file_path)
            except OSError as e:
                print(f"[WARNING] Failed to move existing file to old_dataset: {e}")
        face_roi = frame_gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        cv2.imwrite(file_path, face_roi)



    cv2.imshow('FaceCapture', frame)
    key = cv2.waitKey(100) & 0xff
    if key == 27:  # ESCAPE key
        break
    elif key == 113:  # q key
        break
    elif count >= COUNT_LIMIT: # Take COUNT_LIMIT face samples and stop video capture
        break

print("\n [INFO] Exiting Program and cleaning up stuff")
cam.release()
cv2.destroyAllWindows()
