from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
from src.config import TEXT_DETECTION_CONFIG, OCR_CONFIG
import time
import pytesseract
from src.utils import crop_img, post_process_text
import tensorflow as tf

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class TextDetectionModel:
    def __init__(self):
        self.east_path = TEXT_DETECTION_CONFIG['east_path']
        self.min_confidence = TEXT_DETECTION_CONFIG['min_confidence']
        self.east_width = TEXT_DETECTION_CONFIG['east_width']
        self.east_height = TEXT_DETECTION_CONFIG['east_height']
        self.east_model = cv2.dnn.readNet(self.east_path)
    def text_detection_correction(self, **kwarg):
        #startX correction
        if kwarg['startX']<0:
            startX=0
        else:
            startX = kwarg['startX']
        #startY correction
        if kwarg['startY']<0:
            startY=0
        else:
            startY = kwarg['startY']
        #endX correction
        if kwarg['endX']>kwarg['W']:
            endX=0
        else:
            endX = kwarg['endX']
        #endY correction
        if kwarg['endY']>kwarg['H']:
            endY=0
        else:
            endY = kwarg['endY']
        return (startX, startY, endX, endY)
    def text_detection(self, image):
        start = time.time()
        # load the input image and grab the image dimensions
        orig = image.copy()
        (H, W) = image.shape[:2]
        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (self.east_width, self.east_height)
        rW = W / float(newW)
        rH = H / float(newH)
        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]
        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.east_model.setInput(blob)
        (scores, geometry) = self.east_model.forward(layerNames)
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < self.min_confidence:
                    continue
                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                # add the bounding box coordinates and probability score to
                # our respective lists
                (startX, startY, endX, endY) = self.text_detection_correction(
                    startX=startX, startY=startY, endX=endX, endY=endY, H=H, W=W
                )
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        orig_boxes = []
        for (startX, startY, endX, endY) in (boxes):
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            orig_boxes.append((startX, startY, endX, endY))
        end = time.time()
        # show timing information on text prediction
        print("[INFO] text detection took {:.6f} seconds".format(end - start))
        return orig_boxes

class OCRModel:
    def __init__(self, text_detection_model):
        self.model_type = OCR_CONFIG['model_type']
        self.text_detection_model = text_detection_model
        if self.model_type=='pytesseract':
            self.tesseract_config = OCR_CONFIG['tesseract_config']
        else:
            self.tf_model_path = OCR_CONFIG['tf_model_path']
            self.tf_model = tf.keras.models.load_model(self.tf_model_path, compile=False)
            self.tf_model_config = OCR_CONFIG['tf_model_config']
            self.shape = self.tf_model_config['img_shape']
    def ocr_with_tesseract(self, img):
        start = time.time()
        text_detection_boxes =  self.text_detection_model.text_detection(img)
        texts=[]
        for (startX, startY, endX, endY) in (text_detection_boxes):
            try:
                text = pytesseract.image_to_string(img[startY:endY, startX:endX], config=self.tesseract_config).split()[0]
                text = post_process_text(text)
                texts.append(text)
            except:
                texts.append("")
        texts = " ".join(texts)
        end = time.time()
        print(f'[INFO] OCR with tesseract took {end-start} seconds')
        return texts
    def read_img_and_resize(self, img):
        if self.shape[1] is None:
            img_shape = tf.shape(img)
            scale_factor = self.shape[0] / img_shape[0]
            img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
            img_width = tf.cast(img_width, tf.int32)
        else:
            img_width = self.shape[1]
        img = tf.image.resize(img, (self.shape[0], img_width))
        return img
    def tf_model_inference(self, img):
        outputs = self.tf_model(img)
        text, prob = outputs[0].numpy()[0].decode("utf-8") , outputs[1].numpy()[0]
        if prob<0.5:
            text = ""
        return text
    def ocr_with_crnn(self, img):
        start = time.time()
        text_detection_boxes =  self.text_detection_model.text_detection(img)
        texts = []
        for box in text_detection_boxes:
            cropped_img = crop_img(img, box)
            cropped_img = self.read_img_and_resize(cropped_img)
            cropped_img = tf.expand_dims(cropped_img, 0)
            text = self.tf_model_inference(cropped_img)
            text = post_process_text(text)
            texts.append(text)
        end = time.time()
        print(f'OCR with CRNN took {end-start} seconds')
        return " ".join(texts)
    