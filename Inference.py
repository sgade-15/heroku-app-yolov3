import torch
import numpy as np
import cv2
import json

from general import get_tensor, get_model
from matplotlib import pyplot as plt
from PIL import Image

with open(r'name_to_cat.json') as f:
    name_to_cat = json.load(f)
cat_to_name = {v: k for k, v in name_to_cat.items()}

def get_plant_disease(image_bytes):
    model = get_model()
    tensor=get_tensor(image_bytes)
    outputs = model(tensor)
    _, predicted = torch.max(outputs, 1)
    category = predicted.item()
    disease_name = cat_to_name[category]
    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(outputs) 
    probs = probabilities.detach().numpy()          #1D-array and detacting grad
    top1_prob=probs[0][predicted[0]]
    top1_prob = np.around(top1_prob, decimals=3)
    #print('Probability:   ',top1_prob) #Converted to probabilities
    top3_prob,top3_label = torch.topk(probabilities,3)
    top3_label=top3_label[0].detach().numpy()
    top3_prob=top3_prob[0].detach().numpy()
    top3_disease = [0,0,0]
    for i in range(len(top3_label)):
        top3_disease[i] = cat_to_name[top3_label[i]]
    #print(top3_disease)
    top3_prob = np.around(top3_prob, decimals=3)
    #print(top3_prob)
    return top1_prob, disease_name, top3_disease, top3_prob

def object_detection(image_bytes,val):
    # Load Yolo
    net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # Loading image
    img = image_bytes
    height, width, channels = img.shape
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    #height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    #Informations
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2) 
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    if(len(boxes)==0):
        crop_img=img
        val = 0
    else:
        i = confidences.index(max(confidences)) 
        #print(boxes)
        x0, y0, w, h = boxes[i]
        x1=x0+w
        y1=y0+h
        if (x0<0):
            x0=0
        if(x0+w>=width):
            x1=width-1
        if (y0<0):
            y0=0
        if(y0+h>=height):
            y1=height-1
        crop_img = img[y0:y1, x0:x1]
        val=1
    return(crop_img,val)

def background_removal(image_bytes):
    #cv2.imwrite('static/leaf.png',image_bytes)
    src = image_bytes
    #src = get_tensor(image_bytes)
    #src = src.numpy()[:, :, :]
    #src = src.transpose(2,0,1)
    #src = src.permute(1,2,0)


    #src1 = np.ascontiguousarray(src, dtype=np.uint8)
    #src2 = np.ascontiguousarray(src, dtype=np.float)

    hsv = cv2.bilateralFilter(src,15,50,50)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (18, 70, 0), (86, 225,255))


    ## slice the green
    imask = mask>0
    green = np.zeros_like(src, np.uint8)
    green[imask] = src[imask]

    blurred = cv2.GaussianBlur(green, (3, 3),0)

    blurred_float = blurred.astype(np.float32) / 255.0
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection(r'model.yml/model.yml')
    edges = edgeDetector.detectEdges(blurred_float) * 255.0

    edges_8u = np.asarray(edges, np.uint8)
    edges_8u = cv2.medianBlur(edges_8u, 3)

    def findSignificantContour(edgeImg):
        contours, hierarchy = cv2.findContours(
            edgeImg,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea)    # Ascending Order
        largestContour = contours[-1]                       # -1 => Largest Area
        return largestContour
    
    contour = findSignificantContour(edges_8u)
    # Draw the contour on the original image
    contourImg = np.copy(src)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)

    mask = np.zeros_like(edges_8u)
    cv2.fillPoly(mask, [contour], 255)

    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

    # mark inital mask as "probably background"
    # and mapFg as sure foreground
    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    cv2.grabCut(src, trimap, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # create mask again
    mask2 = np.where(
        (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),
        255,
        0
        ).astype('uint8')

    contour2 = findSignificantContour(mask2)
    mask3 = np.zeros_like(mask2)
    cv2.fillPoly(mask3, [contour2], 255)

    foreground = np.copy(src).astype(float)
    foreground[mask3 == 0] = 255    
    #cv2.imwrite('static/leaf_foreground.png',foreground)
    pil_cutout=Image.fromarray(cv2.cvtColor(foreground.astype('uint8'), cv2.COLOR_BGR2RGB))
    
    return(pil_cutout)
