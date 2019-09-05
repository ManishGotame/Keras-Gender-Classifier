# fixed unecessary crashing of the classifier
# added an controller to resize the video output size
vidName = "gbsvideo.mp4"


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

loaded_model = load_model('MF_weight_70Pixels.h5')
Face_cascade = cv2.CascadeClassifier('Face_Detection/haarcascade_frontalface_default.xml')
img_height = 70
img_width = img_height
# img_width = 50

loaded_model.compile(
		loss='binary_crossentropy', # classify between 2 images, Maybe 0 and 1 
		optimizer='adam', # optimizer, Was supposed to use 'rmsprop' but this seems to work better
		metrics=['accuracy']    
	)

def nothing(a):
	pass

cv2.namedWindow("WINDOW")
cv2.createTrackbar('gray', "WINDOW", 450 , 1300, nothing)

# cap = cv2.VideoCapture('Test_video/splice') # if you want to run this program on a video file

#cap = cv2.VideoCapture(0)
# new changes for the demo 

cap = cv2.VideoCapture(vidName)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('videopy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))

# ends here



try:
	q = Queue()
	z = Queue()
	y = Queue()
	f = Queue()
	xw = Queue()
	yw = Queue()
except:
	q = Queue.Queue()
	z = Queue.Queue()
	y = Queue.Queue()
	f = Queue.Queue()
	xw = Queue.Queue()
	yw = Queue.Queue()
	
	
def Main():
	while(True):
		def Classifier(face_img):
			face_img_matrix = ((cv2.resize(face_img, (img_width,img_height))).astype('float32'))/ 255
			face_img_matrix = face_img_matrix.reshape(1,1,img_width,img_height) # (1,150,150)

			score = ((loaded_model.predict(face_img_matrix)))
			data = (loaded_model.predict_classes(face_img_matrix))
			print(score) 
			y.put(score)
			return data

		def Face_detection(frame):
			try:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				faces = Face_cascade.detectMultiScale(gray, 1.2, 10)
				for (x, y, w, h) in faces: 
					cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),1) # light green, Beautiful !

				cropped_face = frame[y:y+h, x:x+w]
				failed = "Working"
				f.put(failed)
				z.put(cropped_face)
				q.put(frame)
				xw.put(x)
				yw.put(y)
			except Exception as e:
				#print(e)  # it prints the existing error
				failed = "Failed"
				f.put(failed)



		ret, frame = cap.read()
		if ret == True:
			# try:
			frame = cv2.flip(frame, 2) # uncomment this line of code when using webcam
			#frame = imutils.resize(frame, width=550)  # Uncomment this line of code to make the output faster. Increase in speed but decrease in accuracy

			Face_detection(frame)
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
			
			out.write(frame)
			
			# cv2.imshow("Detected Frame",frame)
			if cv2.waitKey(1) == ord('q'):
				break

		elif ret == False:
			break
Main()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

