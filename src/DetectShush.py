import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys
from plotly.api.v2.grids import row

# |----------------------------------------------------------------------------|
# detectShush
# |----------------------------------------------------------------------------|

def detectShush(frame, location, ROI, cascade):
    scaleFactor = 1.50
    neighbors = 5
    flag = 0|cv2.CASCADE_SCALE_IMAGE
    minSize = (10,10)
    row, col = ROI.shape

    cv2.imshow("ROI", ROI)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    mouths = cascade.detectMultiScale(ROI, scaleFactor, neighbors, flag, minSize) 
    mouths = check_box_in_box(mouths)

    for (mx, my, mw,  mh) in mouths:
        mx += location[0]
        my += location[1]
        cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
    return len(mouths) == 0
        
# |--------------------------------detectShush---------------------------------|
    
# |----------------------------------------------------------------------------|
# detect
# |----------------------------------------------------------------------------|

def detect(frame, faceCascade, mouthsCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

#     gray_frame = cv2.equalizeHist(gray_frame)
#     gray_frame = cv2.medianBlur(gray_frame, 5)

    scaleFactor = 1.24
    neighbors = 4
    flag =  0
#     flag =  0|cv2.CASCADE_SCALE_IMAGE
    minSize =  (40, 40)

    faces = faceCascade.detectMultiScale(gray_frame, scaleFactor, neighbors, flag, minSize)
    detected = 0
    
    if len(faces)==0:
        print("No face has been found")
        faceROI = gray_frame
        if detectShush(frame, (0, 0), faceROI, mouthsCascade):
            detected += 1
            print("shush detected")
    else:
        print("New image")
        faces = check_box_in_box(faces)
    
    
    
    for (x, y, w, h) in faces:
        # ROI for mouth
        x1 = x
        h2 = int(h/2)
        y1 = y + h2
        mouthROI = gray_frame[y1:y1+h2, x1:x1+w]

        if detectShush(frame, (x1, y1), mouthROI, mouthsCascade):
            detected += 1
            print("shush detected")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
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



            if not(box1.tolist() in insideBoxList):
                if x1<x2+3 and y1<y2+3 and xx1>xx2-3 and yy1>yy2-3:
                    insideBoxList.append(box2.tolist())
                #if -ends
            #if not -ends
        #for index2 -ends
    #for index1 -ends

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
    files =  [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]
    windowName = None
    totalCnt = 0
    for f in files:
        img = cv2.imread(f)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2)
            totalCnt += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCnt
        
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
    showframe = True
    while(showframe):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            break
        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showframe = False
    
    videocapture.release()
    cv2.destroyAllWindows()
        
# |--------------------------------runonVideo---------------------------------|
    



if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1 +
        "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('./Mouth.xml')
    if(len(sys.argv) == 2): # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, mouth_cascade, folderName)
        print("Total of ", detections, "detections")
    else: # no arguments
        runonVideo(face_cascade, mouth_cascade)