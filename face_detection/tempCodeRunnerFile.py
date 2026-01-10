BASE_DIR = os.path.dirname(__file__)
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier(CASCADE_PATH)