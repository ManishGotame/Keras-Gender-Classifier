# fixed unecessary crashing of the classifier
# added an controller to resize the video output size

import numpy as np
import cv2
import imutils
try:
    import Queue 
except:
    from multiprocessing import Queue
import keras 
from keras.models import load_model
import os 

# ================== classifier weights ====================
# loaded_model = load_model('pretrained_weights/casy_test.h5')
# loaded_model = load_model('pretrained_weights/Male-And-Female-50pixels.h5')
# loaded_model = load_model('pretrained_weights/casy_test_900.h5')
# loaded_model = load_model('pretrained_weights/male_and_female_100pixels_improved_50epochs.h5')
# loaded_model = load_model('pretrained_weights/Male-And-Female-70pixels-1100+500_epochs.h5.h5')
# loaded_model = load_model('pretrained_weights/Male-And-Female-70pixels-1100+500+500_epochs.h5')
loaded_model = load_model('Weights/Male-And-Female-70pixels-1100+500+500+50+50_epochs.h5')
# loaded_model = load_model('pretrained_weights/Sample_weights_810_samples.h5')

Face_cascade = cv2.CascadeClassifier('Face_Detection/haarcascade_frontalface_default.xml')
# ends here

img_height = 70
img_width = img_height
# img_width = 50

loaded_model.compile(
		loss='binary_crossentropy', # classify between 2 images, Maybe 0 and 1 
		#loss='categorical_crossentropy'#categorical_crossentropy classifies between more than 2 images 
		optimizer='adam', # optimizer, Was supposed to use 'rmsprop' but this seems to work better
		metrics=['accuracy']    
	)
# Pool = mp.Pool()
# pool = ThreadPool(2)

def nothing(a):
	pass

cv2.namedWindow("WINDOW")
cv2.createTrackbar('gray', "WINDOW", 450 , 1300, nothing)

# cap = cv2.VideoCapture('Test_video/splice')

cap = cv2.VideoCapture(0)
# cap.set(cv2.cv.CV_CAP_PROP_FPS, 120)

q = Queue()
z = Queue()
y = Queue()
f = Queue()
xw = Queue()
yw = Queue()

def Main():
	while(True):
		def Classifier(face_img):
			face_img_matrix = ((cv2.resize(face_img, (img_width,img_height))).astype('float32'))/ 255
			# face_img_matrix /=255 # decrese the size
			face_img_matrix = face_img_matrix.reshape(1,1,img_width,img_height) # (1,150,150)

			score = ((loaded_model.predict(face_img_matrix)))
			data = (loaded_model.predict_classes(face_img_matrix))
			print(score) 
			y.put(score)
			return data
			# print data

		def Face_detection(frame):
			try:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				faces = Face_cascade.detectMultiScale(gray, 1.2, 10)
				# faces = Face_cascade.detectMultiScale(gray, 5, 10)
				for (x, y, w, h) in faces: 
					cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),1) # light green, Beautiful !

				cropped_face = frame[y:y+h, x:x+w]
				# return frame
				failed = "Working"
				f.put(failed)
				z.put(cropped_face)
				q.put(frame)
				xw.put(x)
				yw.put(y)
			except Exception as e:
				#print e  # it prints the existing error
				failed = "Failed"
				f.put(failed)



		ret, frame = cap.read()
		if ret == True:
			# try:
			frame = cv2.flip(frame, 2) # uncomment this line of code when using webcam
			#frame = imutils.resize(frame, width=550)  # Uncomment this line of code to make the output faster. Increase in speed but decrease in accuracy


			Face_detection(frame)

			# test1 = pool.apply_async(Face_detection, (frame,)) # this one didn't work out well
			failed_result = f.get()
			if failed_result == "Failed":
				pass # Face Not Detected
			elif failed_result == "Working":
				frame = q.get()
				cropped = z.get()
				xh = xw.get()
				yh = yw.get()
				cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
				data = Classifier(cropped)
				score = y.get()

				# frame = imutils.resize(frame, width=video_size)
				font = cv2.FONT_HERSHEY_DUPLEX
				if data == np.array([0]):
					cv2.putText(frame,'Male',(xh-60,yh+25), font,0.7,(0,255,0),1)

				elif data == np.array([1]):
					cv2.putText(frame,'Female',(xh-80,yh+25), font, 0.7,(0,255,0),1)

			video_size = cv2.getTrackbarPos('gray', "WINDOW")
			frame = imutils.resize(frame, width=video_size)

			cv2.imshow("Detected Frame",frame)
			if cv2.waitKey(1) == ord('q'):
				break

		elif ret == False:
			break
Main()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

