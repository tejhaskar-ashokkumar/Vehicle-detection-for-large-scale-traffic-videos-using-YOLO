# import the necessary packages
import numpy as np
import time
import cv2
import os
# construct the argument parse and parse the arguments

YOLO_PATH = 'yolo-coco'
inputPath = 'videos/video.mp4'
outputPath = 'output/output.mp4'

labelsPath = os.path.sep.join([YOLO_PATH, "coco.names"])

LABELS = open(labelsPath).read().strip().split("\n")


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Load YOLO Weights and Model Conf.
weightsPath = os.path.sep.join([YOLO_PATH, "yolov3.weights"])
configPath = os.path.sep.join([YOLO_PATH, "yolov3.cfg"])

print("Loading pre-trained YOLO model . . . .")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


vs = cv2.VideoCapture(inputPath)
writer = None
(W, H) = (None, None)

	
while True:
	
	(grabbed, frame) = vs.read()
	#reaching end of stream
	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]



	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			#extract class ID and confidence
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			
            #filtering least probabilities
			if confidence > 0.5:
				
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				
	# NMS to eliminate weak detections
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
		0.3)
	# ensure at least one detection exists
	if len(idxs) > 0:

		var1 = 0
		var = []
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			cv2.line(frame, (415, 361), (615, 361), (0, 0, 0xFF), 3)
			cv2.line(frame, (670, 361), (843, 361), (0, 0, 0xFF), 3)
			freq1 = [j for j in classIDs]
			
			freq = dict([LABELS[x], classIDs.count(x)] for x in set(classIDs))
			
			freq = str(freq)[1:-1]
			text1 = ("Total Vehicle Count = {}".format(freq))
			cv2.putText(frame, text1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 2)
				
	if writer is None:
		# writing video
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(outputPath, fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	# save the output video
	writer.write(frame)
# release the file pointers
print("Done . . . .")