import cv2
import os
import numpy as np
import time

# Using LBPH(Local Binary Patterns Histograms) recognizer
recognizer=cv2.face.LBPHFaceRecognizer_create()
BASE_DIR = os.path.dirname(__file__)
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier(CASCADE_PATH)
if face_detector.empty():
    print(f"\n[ERROR] Could not load cascade classifier at {CASCADE_PATH}")
    raise SystemExit(1)
path='face_detection\dataset'
path = os.path.join(BASE_DIR, 'dataset')
path = os.path.join(BASE_DIR, 'dataset')
# function to read the images in the dataset, convert them to grayscale values, return samples
def getImagesAndLabels(path):
    faceSamples=[]
    ids = []

    for file_name in os.listdir(path):
        if file_name.endswith(".jpg"):
            id = int(file_name.split(".")[1])
            img_path = os.path.join(path, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            faces = face_detector.detectMultiScale(img)

            for (x, y, w, h) in faces:
                faceSamples.append(img[y:y+h, x:x+w])
                ids.append(id)

    return faceSamples, ids


def trainRecognizer(faces, ids):
    recognizer.train(faces, np.array(ids))
    # Create the 'trainer' folder if it doesn't exist
    trainer_dir = os.path.join(BASE_DIR, 'trainer')
    if not os.path.exists(trainer_dir):
        os.makedirs(trainer_dir)
    # Save the model into 'trainer/trainer.yml'
    model_path = os.path.join(trainer_dir, 'trainer.yml')
    recognizer.write(model_path)

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
# Get face samples and their corresponding labels
time_load_info_start = time.time()
faces, ids = getImagesAndLabels(path)
time_load_info_end = time.time()
print(f"\n [INFO] Loading images and labels took {time_load_info_end - time_load_info_start:.2f} seconds.")

#Train the LBPH recognizer using the face samples and their corresponding labels
time_train_start = time.time()
trainRecognizer(faces, ids)
time_train_end = time.time()
print(f"\n [INFO] Training the recognizer took {time_train_end - time_train_start:.2f} seconds.")


# Print the number of unique faces trained
num_faces_trained = len(set(ids))
total_time = (time_load_info_end - time_load_info_start) + (time_train_end - time_train_start)
print(f"\n [INFO] Total time taken: {total_time:.2f} seconds.")
print("\n [INFO] {} faces trained. Exiting Program".format(num_faces_trained))
