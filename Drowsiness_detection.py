from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound as ws

frequency =5600
duration = 1000

def eyeAspectRatio(eye):
    
    # vertical co-ordinate
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # horizontal co-ordinate
    C = dist.euclidean(eye[0], eye[3])
    ear = (A+B)/(2.0*C)

    return ear

count = 0
earThresh = 0.3 #0.3
print("put 24 for 1 second and 48 for 2 second and so on...")
print(" 1. Hour 2. Minute 3. Second ")
time_select = int(input())
#hour calculation
if (time_select == 1):
    h_time = int(input("Enter Duration : "))
    ef = h_time * 3600 * 24
# minute calc    
elif (time_select == 2):
    m_time = int(input("Enter Duration : "))
    ef = m_time * 60 * 24
#second calc
elif (time_select == 3):
    s_time = int(input("Enter Duration : "))
    ef = s_time * 24
else:
    ef = 24
    

print(ef) 
#earFrames = int(input("Enter Duration : "))  
shapePredictor = "shape_predictor_68_face_landmarks.dat"

cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=650)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (239, 19, 232), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (239, 19, 232), 1)
        

        if ear < earThresh:
            count += 1

            if count >= ef:
                cv2.putText(frame, "DROWSINESS DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 20, 255), 2)
                ws.Beep(frequency, duration)

        else:
            count = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0XFF

    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
        






















                                 
