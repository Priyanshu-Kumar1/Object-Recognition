import cv2


cap= cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


config_file= "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model= "frozen_inference_graph.pb"

model= cv2.dnn_DetectionModel(frozen_model, config_file)



classLabels= []
file_name= "Labels.txt"
with open(file_name, "rt") as fpt:
    classLabels= fpt.read().rstrip("\n").split("\n")


model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


while True:
    success, img= cap.read()

    classIds, confs, bbox= model.detect(img, confThreshold= .5)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0,102,255), thickness= 3)
            try:
                cv2.putText(img, classLabels[classId-1], (box[0]+10, box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,102,255),2)

            except:
                print("Nothing Here")
    
    cv2.imshow("OutPut", img)
    cv2.waitKey(1)

