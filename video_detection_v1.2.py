import numpy as np
import cv2
import imutils
# import thread
import time
import Queue 
# import multiprocessing as mp 
# from multiprocessing import Process
# import threading
import thread
# from multiprocessing.pool import ThreadPool
import keras 
from keras.models import load_model

# from multiprocessing.dummy import Pool as ThreadPool

loaded_model = load_model('Male-And-Female-100pixels.h5')
loaded_model.compile(
		loss='binary_crossentropy', # classify between 2 images, Maybe 0 and 1 
		#loss='categorical_crossentropy'#categorical_crossentropy classifies between more than 2 images 
		optimizer='adam', # optimizer, Was supposed to use 'rmsprop' but this seems to work better
		metrics=['accuracy']    
	)
# Pool = mp.Pool()
# pool = ThreadPool(2)

cap = cv2.VideoCapture(0)
# cap.set(cv2.cv.CV_CAP_PROP_FPS, 120)

Face_cascade = cv2.CascadeClassifier('Face_Detection/haarcascade_frontalface_default.xml')
# Body_cascade = cv2.CascadeClassifier('Face_Detection/haarcascade_fullbody.xml')
# print(Face_cascade)
# pool = ThreadPool(processes=2)
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
			# face_img_matrix = face_img_matrix.reshape(face_img_matrix.shape[0],1,img_height,img_width)# (1,1,150,150)
			# print(face_img.shape)

			#model prediction module
			# loaded_model = load_model('Graph/male_and_female.h5')
			# print face_img_matrix
			score = ((loaded_model.predict(face_img_matrix)))
			data = (loaded_model.predict_classes(face_img_matrix))
			y.put(score)
			return data
			# print score 
			# print data

		def Face_detection(frame):
			try:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				faces = Face_cascade.detectMultiScale(gray, 1.2, 5)
				# faces = Face_cascade.detectMultiScale(gray, 10, 10)
				for (x, y, w, h) in faces: 
					cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
					# cv2.ellipse(frame,(x,y),(x+w,y+h),0,0,180,255,-1)
					# cv2.circle(frame,(x,y), (w+h), (0,0,255), -1)

				# cropped_face = frame[x:x+h,y:y+w]
				cropped_face = frame[y:y+h, x:x+w]
				# return frame
				z.put(cropped_face)
				q.put(frame)
				# time.sleep(timer)
			except Exception as e:
				print e
				# pass
				cropped_face = frame[20:20+40,4:4+2]
				# return cropped_face
				# return frame
				z.put(cropped_face)
				q.put(frame)
				# time.sleep(timer)
				# Capture frame-by-frame



		ret, frame = cap.read()
		if ret == True:
			try:
				frame = cv2.flip(frame, 1)
				# frame = imutils.resize(frame, width=450)

				Face_detection(frame)
				
				# thread.start_new_thread(Face_detection, (frame,))


				
				# task = pool.apply_async(Face_detection, (frame,))
				# task.close()    

				# test1 = pool.apply_async(Face_detection, (frame,))
				frame = q.get()
				cropped = z.get()

				data = Classifier(cropped)

				score = y.get()
				# test2 = pool.apply_async(Classifier, (cropped,))

				# Thread1 = threading.Thread(target=Classifier, args=[cropped])
				# Thread1.start()
				# Thread1.join()
			   
				# Classifier(cropped)
				# pool.map(Classifier, cropped)
				# pool.close()
				# pool.join()

				# t = threading.Thread(target=Classifier, args=(cropped,))
				# t.daemon = True
				# t.start()

				# p = Process(target=Classifier, args=(cropped,))
				# p.start()
				# p.join()
				
				# test2 = pool.apply_async(Classifier, (cropped,))


				# frame = imutils.resize(frame, width=450)
				font = cv2.FONT_HERSHEY_SIMPLEX
				if data == np.array([0]):
					cv2.putText(frame,'Male',(10,50), font, 1,(255,255,255),2)
					cv2.putText(frame, str(score[0]),(90,50), font, 1,(255,255,255),2)

				elif data == np.array([1]):
					cv2.putText(frame,'Female',(10,50), font, 1,(255,255,255),2)
					cv2.putText(frame, str(score[1]),(150,50), font, 1,(255,255,255),2)
				cv2.imshow("Frame",frame)
				# cv2.imshow("Cropped",hello)

				if cv2.waitKey(1) == ord('q'):
					break

			except Exception as e:
				print e 
Main()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# from multiprocessing import Process
# #omitted code
# import cv2

# cap = cv2.VideoCapture(0)

# while True:
#     # getting tick count
#     e1 = cv2.getTickCount()
#     # storing frame
#     _, img = cap.read()
#     # extract grid - first subsystem
#     # gridExtractor.extractGrid(img)
#     # define red colours in the screen - second subsystem
#     # findRedColours(img, board)
#     # getting tick count after the functions
#     e2 = cv2.getTickCount()
#     # calculating time
#     t = (e2 - e1) / cv2.getTickFrequency()
#     # print time
#     cv2.imshow("a",img)
#     print(t)

#     # check if img is none
#     if img is not None:
#         # omitted code

#         k = cv2.waitKey(20) & 0xFF
#         # start the game, hide info
#         if (k == ord('s') or k == ord('S')) and start is False:
#             # create new thread to play game
#             p = Process(target=playGame)
#             p.start()
