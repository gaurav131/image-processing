import dlib
import cv2
cam = cv2.VideoCapture(0)
face_detector = dlib.cnn_face_detection_model_v1("human_face_detector.dat")
while True:
    _, img = cam.read()
    img = cv2.flip(img, 1)
    black = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets  = face_detector(black)
    for loc in dets:
        face = loc.rect
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        cv2.rectangle(img,  (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow("img", img)
    if cv2.waitKey(5) == 27:
        cv2.destroyAllWindows()
        break