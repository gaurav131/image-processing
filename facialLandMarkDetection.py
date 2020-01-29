import dlib
from imutils import face_utils
import cv2
import time
colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)]
shape = dlib.shape_predictor("faceShape.dat")
cam = cv2.VideoCapture(0)
face_detector = dlib.cnn_face_detection_model_v1("human_face_detector.dat")
count = 0
while True:
    start = time.time()
    try:
        _, img = cam.read()
        count += 1
        img = cv2.flip(img, 1)
        black = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets  = face_detector(black)
        for loc in dets:
            face = loc.rect
            geom = shape(black, face)
            geometry = face_utils.shape_to_np(geom)
            landmarks = face_utils.FACIAL_LANDMARKS_68_IDXS
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
            left_eye_contours = geometry[left_eye[0]:left_eye[1]]
            right_eye_contours = geometry[right_eye[0]:right_eye[1]]
            left_eye_contours = cv2.convexHull(left_eye_contours)
            right_eye_contours = cv2.convexHull(right_eye_contours)
            cv2.drawContours(img,left_eye_contours,-1,[0,0,255], 5)
            cv2.drawContours(img,right_eye_contours,-1,[0,0,255], 5)
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            cv2.rectangle(img,  (x,y), (x+w,y+h), (0,255,0), 2)
            img = face_utils.visualize_facial_landmarks(img, geometry, colors=colors)
    except:
        pass
    end = time.time()
    fps = int(count/(end-start))
    count = 0
    cv2.putText(img, "FPS:- "+str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("img", img)
    if cv2.waitKey(5) == 27:
        cv2.destroyAllWindows()
        break
