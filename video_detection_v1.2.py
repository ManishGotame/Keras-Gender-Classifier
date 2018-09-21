import numpy as np
import cv2
import imutils
# import thread
import time
import Queue 

import thread
# from multiprocessing.pool import ThreadPool
import keras 
from keras.models import load_model

# from multiprocessing.dummy import Pool as ThreadPool

loaded_model = load_model('Male-And-Female-100pixels.h5')
loaded_model.compile(
		loss='binary_crossentropy', # classify between 2 images, Maybe 0 and 1 
		optimizer='adam', # optimizer, Was supposed to use 'rmsprop' but this seems to work better
		metrics=['accuracy']    
	)

cap = cv2.VideoCapture(0)
# cap.set(cv2.cv.CV_CAP_PROP_FPS, 120)

Face_cascade = cv2.CascadeClassifier('Face_Detection/haarcascade_frontalface_default.xml')
q = Queue.Queue()
z = Queue.Queue()
y = Queue.Queue()
def Main():

# y = Queue.Queue()   
	# def Classification():

	while(True):
		def Classifier(face_img):
			img_height = 100
			img_width = 100
			face_img = cv2.resize(face_img, (img_height,img_width)) #convert into 150,150 size
			face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) # convert rgb to gray because the neural
			# net is built to learn from gray images
			face_img_matrix = np.array(face_img) # convert into a numpy array 
			face_img_matrix = face_img_matrix.astype('float32')
			face_img_matrix /=255 # decrese the size
			face_img_matrix = face_img_matrix.reshape(1,1,img_width,img_height) # (1,150,150)
			score = ((loaded_model.predict(face_img_matrix)))
			data = (loaded_model.predict_classes(face_img_matrix))
			y.put(score)
			return data

		def Face_detection(frame):
			try:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				faces = Face_cascade.detectMultiScale(gray, 1.2, 5)
				# faces = Face_cascade.detectMultiScale(gray, 10, 10)
				for (x, y, w, h) in faces: 
					cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

				cropped_face = frame[y:y+h, x:x+w]
				# return frame
				z.put(cropped_face)
				q.put(frame)
				# time.sleep(timer)
			except Exception as e:
				print e
				# pass
				cropped_face = frame[20:20+40,4:4+2]
				z.put(cropped_face)
				q.put(frame)

		ret, frame = cap.read()
		if ret == True:
			try:
				frame = cv2.flip(frame, 1)
				# frame = imutils.resize(frame, width=450)

				Face_detection(frame)
				frame = q.get()
				cropped = z.get()

				data = Classifier(cropped)

				score = y.get()
				font = cv2.FONT_HERSHEY_SIMPLEX
				if data == np.array([0]):
					cv2.putText(frame,'Male',(10,50), font, 1,(255,255,255),2)
					cv2.putText(frame, str(score[0]),(90,50), font, 1,(255,255,255),2)

				elif data == np.array([1]):
					cv2.putText(frame,'Female',(10,50), font, 1,(255,255,255),2)
					cv2.putText(frame, str(score[1]),(150,50), font, 1,(255,255,255),2)
				cv2.imshow("Frame",frame)

				if cv2.waitKey(1) == ord('q'):
					break

			except Exception as e:
				print e 
Main()

cap.release()
cv2.destroyAllWindows()
