import cv2
import dlib
from scipy.spatial import distance
from playsound import playsound 

def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

cap = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat.bz2")
count_frame=0

while True:
	_, frame = cap.read()
	frame=cv2.flip(frame,1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_detector(gray)

	for face in faces:
		face_landmarks = dlib_facelandmark(gray, face)
		leftEye = []
		rightEye = []

		for n in range(36, 42):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			leftEye.append((x, y))
			next_point = n + 1
			if n == 41:
				next_point = 36
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame, (x, y), (x2, y2), (255, 0, 0), 1)

		for n in range(42, 48):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			rightEye.append((x, y))
			next_point = n + 1
			if n == 47:
				next_point = 42
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame, (x, y), (x2, y2), (255,0,0), 1)

		left_ear = calculate_EAR(leftEye)
		right_ear = calculate_EAR(rightEye)

		EAR = (left_ear + right_ear) / 2
		EAR = round(EAR, 2)
		cv2.putText(frame,"EAR="+str(EAR), (470, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 4)
		if EAR<0.26:
			count_frame+=1
			if count_frame>=35:
				cv2.putText(frame,"Drowsiness Alert!!",(30,50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),4)
				playsound("annoying-alarm-tone.mp3")
		else:
			count_frame=0


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1)
	if key == 27:
		break
cap.release()
cv2.destroyAllWindows()
