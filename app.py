# Importing required libraries, obviously
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os


def detect_objects(image):
    confidenceThreshold = 0.2
    NMSThreshold = 0.3

    modelConfiguration = 'yolov3_testing.cfg'
    modelWeights = 'yolov3_training_final.weights'

    labelsPath = 'classes.txt'
    labels = open(labelsPath).read().strip().split('\n')

    np.random.seed(10)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    image = np.array(image.convert('RGB'))
    #image = cv2.imread(image)
    (H, W) = image.shape[:2]

    # Determine output layer names
    layerName = net.getLayerNames()
    layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layersOutputs = net.forward(layerName)

    boxes = []
    confidences = []
    classIDs = []
    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    outputs = {}
    # Apply Non Maxima Suppression
    detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)
    if len(detectionNMS) > 0:
        for i in detectionNMS.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


def about():
    st.write('''
    Safety has always been a very important issue in all industrial activities, especially construction and on roads. 
    Since construction involves a large proportion of the workforce, construction fatalities also affect a large population. 
    For instance, in the United States, construction represents 5 to 6% of the workforce but accounts for 15% of 
    work-related fatalities—more than any other sector. The construction sector in Japan is 10% of the workforce 
    but has 42% of the work-related deaths; in Sweden, the numbers are 6% and 13%, respectively.''')


def main():
    st.title("Safety Helmet Detection App : ")
    st.write("**Great Learning Capstone Team 3**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Home":
        st.write("Go to the About section from the sidebar to learn more about it.")

        # You can specify more file types below if you want
        image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg'])

        if image_file is not None:
            image = Image.open(image_file)
            if st.button("Process"):
                result_img = detect_objects(image)
                st.image(result_img, use_column_width=True)
    # st.success("Found {} faces\n".format(len(result_faces)))

    elif choice == "About":
        about()


if __name__ == "__main__":
    main()
