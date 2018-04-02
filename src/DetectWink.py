import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys

# |----------------------------------------------------------------------------|
# detectWink
# |----------------------------------------------------------------------------|
def detectWink(frame, location, ROI, cascade):
    '''
    
    '''
#     ROI = cv2.equalizeHist(ROI)
    scaleFactor = 1.15
    #to increase the reliability (Higher value-> heigher reliability)
    neighbors = 5
    flag = 0|cv2.CASCADE_SCALE_IMAGE
    minSize = (10,20)
    row, col = ROI.shape
    
    newRow  = int(row*3/5)
    ROI = ROI[0:newRow, :]
#     cv2.imshow("ROI", ROI)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    eyes = cascade.detectMultiScale(ROI, scaleFactor, neighbors, flag, minSize) 
    eyes = check_box_in_box(eyes)

    
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
    return len(eyes) == 1    # number of eyes is one
        
# |--------------------------------detectWink---------------------------------|
    
# |----------------------------------------------------------------------------|
# detect
# |----------------------------------------------------------------------------|
def detect(frame, faceCascade, eyesCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # possible frame pre-processing:
#     gray_frame = cv2.equalizeHist(gray_frame)
#     gray_frame = cv2.medianBlur(gray_frame, 7)
#     cv2.imshow("gray frame", gray_frame)


    scaleFactor = 1.25 # range is from 1 to ..
    minNeighbors = 4   # range is from 0 to ..
    flag = 0 # either 0 or 0|cv2.CASCADE_SCALE_IMAGE 
    minSize = (30,30) # range is from (0,0) to ..
 
    faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor, 
        minNeighbors, 
        flag, 
        minSize)
    
#     if len(faces)==0:
#         print("No face has been found")
#     else:
    print("New image")
    faces = check_box_in_box(faces)
     

    detected = 0
    for (x,y,w,h) in faces:
#         x, y, w, h = f[0], f[1], f[2], f[3]
        faceROI = gray_frame[y:y+h, x:x+w]
#         cv2.imshow("Face ROI", faceROI)
#         cv2.waitKey(0)
        if detectWink(frame, (x, y), faceROI, eyesCascade):
            detected += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
    return detected
        
# |--------------------------------detect---------------------------------|


# |----------------------------------------------------------------------------|
# check_box_in_box
# |----------------------------------------------------------------------------|
def check_box_in_box(boxList):
    '''
    check whether the more than 1 object are around the same object and box inside box case
    If yes, take the larger box
    '''

    finalBoxList = []
    insideBoxList = []
    for index1, box1 in enumerate(boxList):
        for index2, box2 in enumerate(boxList):
            if index1==index2:
                continue
            x1,y1,xx1,yy1 = box1[0],box1[1],box1[0]+box1[2],box1[1]+box1[3]
            x2,y2,xx2,yy2 = box2[0],box2[1],box2[0]+box2[2],box2[1]+box2[3]

            # debug
            print("box1 = {}: ({},{},{},{})".format(box1, x1,y1,xx1,yy1))
            print("box2 = {} ({},{},{},{})".format(box2,x2,y2,xx2,yy2))
            
            # debug -ends


            if not(box1.tolist() in insideBoxList):
                if x1<x2+3 and y1<y2+3 and xx1>xx2-3 and yy1>yy2-3:
                    insideBoxList.append(box2.tolist())
                #if -ends
            #if not -ends
        #for index2 -ends
    #for index1 -ends
    # debug
    print("insideBoxList := {}".format(insideBoxList))
    # debug -ends

    for box in boxList:
        if box.tolist() not in insideBoxList:
            finalBoxList.append(box)

    return finalBoxList
# |--------------------------------check_box_in_box---------------------------------|

# |----------------------------------------------------------------------------|
# run_on_folder
# |----------------------------------------------------------------------------| 
def run_on_folder(cascade1, cascade2, folder):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]

    windowName = None
    totalCount = 0
    for f in files:
        img = cv2.imread(f, 1)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2)
            totalCount += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return totalCount
# |--------------------------------run_on_folder---------------------------------|
    
# |----------------------------------------------------------------------------|
# runonVideo
# |----------------------------------------------------------------------------|
def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while(showlive):
        ret, frame = videocapture.read()
 
        if not ret:
            print("Can't capture frame")
            exit()
 
        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False
     
    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()

        
# |--------------------------------runonVideo---------------------------------|
    


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1 
              + "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_eye.xml')
#     big_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "https://github.com/sightmachine/SimpleCV/blob/master/SimpleCV/Features/HaarCascades/two_eyes_big.xml")
    # debug
    print("cv2.data.haarcascades = {}".format(cv2.data.haarcascades))
    # debug -ends


    if(len(sys.argv) == 2): # one argument
        folderName = sys.argv[1]
#         detections = run_on_folder(face_cascade, eye_cascade, folderName)
        detections = run_on_folder(face_cascade, eye_cascade, folderName)

        print("Total of ", detections, "detections")
    else: # no arguments
        runonVideo(face_cascade, eye_cascade)
