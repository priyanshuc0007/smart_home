import dlib
import cv2

# Load the dlib models
def load_dlib_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
    face_rec_model = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
    return detector, predictor, face_rec_model

# Process an image for face detection
def process_image(image_path, detector, predictor, face_rec_model):
    # Load and process the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Face detection
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
    
    return faces, face_descriptor
