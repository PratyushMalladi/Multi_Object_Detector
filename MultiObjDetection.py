import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
listofnames= []
OBJFile = "obj names.names"
with open(OBJFile,"rt") as f:
    listofnames=[line.rstrip() for line in f]

configFILE = "configuration.pbtxt"
HEART = "frozen_inference_graph.pb"

AI = cv2.dnn_DetectionModel(HEART,configFILE)
AI.setInputSize(300,300)
AI.setInputScale(1.0/195)
AI.setInputCrop(215)
AI.setInputMean((104,117,123))
AI.setInputSwapRB(True)

while True:
    somex,pic = vid.read()
    nums, con, frame = AI.detect(pic,confThreshold=0.57)

    if len(nums) != 0:
        for i, j,k in zip(nums.flatten(),con.flatten(),frame):
            cv2.rectangle(pic,k,color=(180,25,45),thickness=4)
            cv2.putText(pic,listofnames[i-1].capitalize(), [k[0]+7, k[1]+35],
            cv2.FONT_HERSHEY_DUPLEX,0.8, (0,0,0), 1)
    cv2.imshow("project",pic)
    cv2.waitKey(2)