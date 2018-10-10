
import numpy as np
import cv2
import imutils
import Queue 
import keras 
from keras.models import load_model

# ================== classifier weights ====================
# loaded_model = load_model('casy_test.h5')
# loaded_model = load_model('Male-And-Female-50pixels.h5')
# loaded_model = load_model('casy_test_900.h5')
# loaded_model = load_model('male_and_female_100pixels_improved_50epochs.h5')
# loaded_model = load_model('Male-And-Female-70pixels-1100+500_epochs.h5.h5')
# loaded_model = load_model('Male-And-Female-70pixels-1100+500+500_epochs.h5')
loaded_model = load_model('Male-And-Female-70pixels-1100+500+500+50+50_epochs.h5')
Face_cascade = cv2.CascadeClassifier('Face_Detection/haarcascade_frontalface_default.xml')
# ends here

img_height = 70
img_width = img_height
# img_width = 50

vid_height = 1280
vid_width = 720 

loaded_model.compile(
		loss='binary_crossentropy', # classify between 2 images, Maybe 0 and 1 
		#loss='categorical_crossentropy'#categorical_crossentropy classifies between more than 2 images 
		optimizer='adam', # optimizer, Was supposed to use 'rmsprop' but this seems to work better
		metrics=['accuracy']    
	)
# Pool = mp.Pool()
# pool = ThreadPool(2)

def GRAB_screen(height, width): #screen cap
		left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
		top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN) 

		hwin = win32gui.GetDesktopWindow()
		hwindc = win32gui.GetWindowDC(hwin)
		srcdc = win32ui.CreateDCFromHandle(hwindc)
		memdc = srcdc.CreateCompatibleDC()
		bmp = win32ui.CreateBitmap()
		bmp.CreateCompatibleBitmap(srcdc, width, height)
		memdc.SelectObject(bmp)
		memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
		# memdc.BitBlt((0, 0), (width, height), srcdc, (width, height), win32con.SRCCOPY)
		
		signedIntsArray = bmp.GetBitmapBits(True)
		img = np.fromstring(signedIntsArray, dtype='uint8')
		img.shape = (height, width,4)

		srcdc.DeleteDC()
		memdc.DeleteDC()
		win32gui.ReleaseDC(hwin, hwindc)
		win32gui.DeleteObject(bmp.GetHandle())

		img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cap = cv2.VideoCapture('female_ijustine.mp4')

# cap = cv2.VideoCapture(0)
# cap.set(cv2.cv.CV_CAP_PROP_FPS, 120)

# Body_cascade = cv2.CascadeClassifier('Face_Detection/haarcascade_fullbody.xml')
# print(Face_cascade)
# pool = ThreadPool(processes=2)
q = Queue.Queue()
z = Queue.Queue()
y = Queue.Queue()
f = Queue.Queue()
d = Queue.Queue()
def Main():
# y = Queue.Queue()   
	# def Classification():

	while(True):
		def Classifier(face_img):
			# face_img_matrix = cv2.resize(face_img, (img_height,img_width)) #convert into 150,150 size
			# face_img_matrix = cv2.cvtColor(face_img_matrix, cv2.COLOR_BGR2GRAY) # convert rgb to gray because the neural
			# net is built to learn from gray images
			# face_img_matrix = np.array(face_img) # convert into a numpy array 
			face_img_matrix = ((cv2.resize(face_img, (img_width,img_height))).astype('float32'))/ 255
			# face_img_matrix /=255 # decrese the size
			face_img_matrix = face_img_matrix.reshape(1,1,img_width,img_height) # (1,150,150)
			# face_img_matrix = face_img_matrix.reshape(face_img_matrix.shape[0],1,img_height,img_width)# (1,1,150,150)
			# print(face_img.shape)

			#model prediction module
			# loaded_model = load_model('Graph/male_and_female.h5')
			# print face_img_matrix
			score = ((loaded_model.predict(face_img_matrix)))
			data = (loaded_model.predict_classes(face_img_matrix))
			print score 
			y.put(score)
			return data
			# print data

		def Face_detection(frame):
			try:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				faces = Face_cascade.detectMultiScale(gray, 1.2, 5)
				# faces = Face_cascade.detectMultiScale(gray, 5, 10)
				for (x, y, w, h) in faces: 
					# cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1) # light green, Beautiful !
					cv2.rectangle(frame,(x,y),(x+w,y+h),(221,221,221),1) # this is even more beautiful
					# cv2.rectangle(frame,(x,y),(x+w,y+h),(128,128,128),1)
					# cv2.ellipse(frame,(x,y),(x+w,y+h),0,0,180,255,-1)
					# cv2.circle(frame,(x,y), (w+h), (0,0,255), -10)

				# cropped_face = frame[x:x+h,y:y+w]
				cropped_face = frame[y:y+h, x:x+w]
				# return frame
				failed = "Working"
				f.put(failed)
				z.put(cropped_face)
				q.put(frame)
			except Exception as e:
				#print e  # it prints the existing error
				failed = "Failed"
				f.put(failed)



		#ret, frame = cap.read()
		frame = GRAB_screen(vid_height, vid_width)
		# if ret == True:
		# try:
		# frame = cv2.flip(frame, 10)
		# frame = imutils.resize(frame, width=450)
		# frame = imutils.resize(frame, width=850)

		Face_detection(frame)

		# test1 = pool.apply_async(Face_detection, (frame,)) # this one didn't work out well
		failed_result = f.get()
		if failed_result == "Failed":
			pass # Face Not Detected
		elif failed_result == "Working":
			frame = q.get()
			cropped = z.get()
			cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
			data = Classifier(cropped)
			score = y.get()

			# frame = imutils.resize(frame, width=450)
			font = cv2.FONT_HERSHEY_SIMPLEX
			if data == np.array([0]):
				cv2.putText(frame,'Male',(10,50), font,1,(0,255,0),2)
				# cv2.putText(frame, str(score),(90,50), font, 1,(255,255,255),2)

			elif data == np.array([1]):
				cv2.putText(frame,'Female',(10,50), font, 1,(0,255,0),2)
				# cv2.putText(frame, str(score),(150,50), font, 1,(255,255,255),2)

		# frame = imutils.resize(frame, width=1000)
		cv2.imshow("qFrame",frame)

		if cv2.waitKey(1) == ord('q'):
			break

		# except Exception as e:
		# 	print e 
Main()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

