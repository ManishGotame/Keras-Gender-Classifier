# Keras-Opencv-Video-Detection

It is a real time Gender Classifier which can classify between whether as human is a Male or a Female. 
I trained a Model which takes a (70,70) resized frame and classifies it as a Male or Female.

Few bugs and Issues were there:
  If the face is not detected, the classifier will take a random cropped frame and try to classify it which is not good.  
  A new trained weight is not used. So, Classification might be wrong
  
Fixed Issue:
  The classifier only starts to classify if the opencv haarcascade finds a face in the video frame.
  A properly trained Model is used and the classification is satisfactory. 
  
 
