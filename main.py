from tkinter import Image
import cv2
import numpy as np
import os
import dlib 
import face_recognition

def faceBox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 4)
    return frame, bboxs


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"



faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']



path= 'images'
images = []
classnames = []
myList = os.listdir(path)
# print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])
# print(classnames)

def findEncodings(images):
    encodlist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodlist.append(encode)
    return encodlist

encodeListKnown = findEncodings(images)
print("Encoding done")



while True:

    video=cv2.VideoCapture(0)

    padding=20

    while True:
        ret,frame=video.read()
        frame,bboxs=faceBox(faceNet,frame)
        for bbox in bboxs:
            face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPred=genderNet.forward()
            gender=genderList[genderPred[0].argmax()]


            ageNet.setInput(blob)
            agePred=ageNet.forward()
            age=ageList[agePred[0].argmax()]


            label="{},{}".format(gender,age)
            cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0), 4) 
            cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
        cv2.imshow("Age-Gender",frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            test = "test_face.png"
            cv2.imwrite(test, frame)
            break

    video.release()

    cv2.destroyAllWindows()

    tstimg = cv2.imread(f'test_face.png')
    tstimg = cv2.cvtColor(tstimg, cv2.COLOR_BGR2RGB)
    encodetstimg = face_recognition.face_encodings(tstimg)[0]
    for en in encodeListKnown:
        matches = face_recognition.compare_faces(encodeListKnown,encodetstimg)
        faceDis = face_recognition.face_distance(encodeListKnown,encodetstimg)
        matchIndex = np.argmin(faceDis)
        
    # print(faceDis[np.argmin(faceDis)])
    if(faceDis[np.argmin(faceDis)]>0.55):
        print("No match found try again")
    else:
        name = classnames[matchIndex].upper()
        print("Name:- ", name)
        print("Gender :- ", gender)
        print("Probable Age range :- ", age)
        break

    var=input("Press 'Q' to exit or 'enter' to continue.")
    if(var=='Q'):
        break;
    else:
        continue
