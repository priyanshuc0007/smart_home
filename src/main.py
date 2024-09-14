import cv2
import fce_recognition as fr_module
import object_detection as od_module

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame from camera")
            break

        frame = fr_module.recognize_faces(frame, fr_module.encodeListKnown, fr_module.personNames)
        frame = od_module.detect_objects(frame)

        cv2.imshow('Object and Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
