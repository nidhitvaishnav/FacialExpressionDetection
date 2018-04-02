#=====================================================================================================================================|
						Facial Expression detection
#=====================================================================================================================================|

Here in given project we are performing two facial expression detection using Viola-Jones approach.
1. Wink Detection
2. Shush detection

-> For object detection, we are creating cascade classifier objects of Haar Cascade using OpenCV.
-> After that, we are detecting object using cascade.detectMultiScale(ROI, scaleFactor, neighbors, flag, minSize).
	-> ROI : the size of the frame from which we wish to detect object
	-> scaleFactor : How much image size is reduced at each image scale
	-> neighbors : number of neighbors each candidate rectangle should retain
	-> flag : cascade type (used to specify some old cascades)
	-> minSize : minimum possible object size (objects smaller than min size are ignored)
#-------------------------------------------------------------------------------------------------------------------------------------|
1. Wink Detection:
#-------------------------------------------------------------------------------------------------------------------------------------|
-> Here, we are first detecting faces of people using face_cascade.
-> If we are able to detect face, than we will check, whether given facebox is inside other facebox(2 faces are detected for the same person). 
	-> If yes, then we will consider bigger (outer) box only.
	-> Now we will detect eyes on the upper 3/5th part of the faces using eye_cascade.
	
-> If face is not detected, then we will detect eyes on the entire image using eye_cascade.
-> Again we will check whether same eye is detected twice, if yes, we will ignore the smaller box.

#-------------------------------------------------------------------------------------------------------------------------------------|
2. Shush Detection:
#-------------------------------------------------------------------------------------------------------------------------------------|
-> Here, we are first detecting faces of people using face_cascade.
-> If we are able to detect face, than we will check, whether given facebox is inside other facebox(2 faces are detected for the same person). 
	-> If yes, then we will consider bigger (outer) box only.
	-> Now we will detect mouth on faces using mouth_cascade.
	
-> If face is not detected, then we will detect mouth on the entire image using mouth_cascade.
-> Again we will check whether same mouth is detected twice, if yes, we will ignore the smaller box.

#-------------------------------------------------------------------------------------------------------------------------------------|
Note:
#-------------------------------------------------------------------------------------------------------------------------------------|
-> Here, face_cascade, eye_cascade, mouth_cascade are provided at http://alereimondo.no-ip.org/OpenCV/34 which are Haar cascade.

-> We are storing output of WinkImg in ../WinkOP
-> We are storing output of ShushImg in ../ShushOP

#=====================================================================================================================================|