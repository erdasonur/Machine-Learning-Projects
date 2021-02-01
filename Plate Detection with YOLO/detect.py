import cv2 as cv
import numpy as np


def main():
    inp_width = 416
    inp_height = 416

    model_configuration = "./darknet-yolov3.cfg"
    model_weights = "./model.weights"
    image = cv.imread('./images/car.jpg')
    plate = image
    net = cv.dnn.readNetFromDarknet(model_configuration, model_weights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)	
    blob = cv.dnn.blobFromImage(image, 1 / 255, (inp_width, inp_height), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    outs = net.forward([net.getLayerNames()[i[0] - 1] for i in net.getUnconnectedOutLayers()])

    image_height = image.shape[0]
    image_width = image.shape[1]

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                width = int(detection[2] * image_width)
                height = int(detection[3] * image_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 3)
        plate = image[top:top + height, left:left + width]

    
    cv.imwrite("./images/plaka.jpg", plate)
main()