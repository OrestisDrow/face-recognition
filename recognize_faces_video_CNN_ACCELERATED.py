# USAGE
# python3 recognize_faces_video_CNN_ACCELERATED.py --encodings encodings.pickle
# --output output/output1.avi --display 1

# import the necessary packages
from imutils.video import VideoStream
from datetime import datetime
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--batchSize", type=int, default=1,
	help="batch size for parrallel proccessing")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

# initialize variables to calculate framerate
num_frames = 0
fps = 0
startTime = datetime.now()
batch_size = args["batchSize"]
# loop over frames from the video file stream
while True:
	# grab (batchSize) frames from the threaded video stream
	frame = []
	rgb = []
	r = []
	for i in range(batch_size):
		frame_ = vs.read()
		rgb_ = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
		rgb_ = imutils.resize(frame_, width=750)
		rgb.append(rgb_)
		r.append(frame_.shape[1] / float(rgb_.shape[1]))
		frame.append(frame_)
		
		time.sleep(0.001)
		'''
		# this code was used to check if we are double sampling 
		# the same image because the I/O buffer did not update
		f = vs.read()
		compare = cv2.compare(frame_, f, 0)
		if compare.all():
			print("____SAME_FRAME____")
		'''

	# parrallel face detection in image batch
	batch_boxes = face_recognition.batch_face_locations(rgb, number_of_times_to_upsample=0)
	#print(len(batch_boxes))
	
	# loop for each image
	for k in range(batch_size):
		# calculating encodings
		encodings = face_recognition.face_encodings(rgb[k], batch_boxes[k])
		names = []
		name = []

		for encoding in encodings:
			# attempt to match each face in the input image to our known
			# encodings
			matches = face_recognition.compare_faces(data["encodings"],
				encoding, tolerance=0.5)
			name = "Unknown"

			# check to see if we have found a match
			if True in matches:
				# find the indexes of all matched faces 
				# then initialize a dictionary to count the
				# total number of times each face was matched
				matchedIdxs = [i for (i, b) in enumerate(matches) if b]
				counts = {}

				# loop over the matched indexes and maintain 
				# a count for each recognized face face
				for i in matchedIdxs:
					name = data["names"][i]
					counts[name] = counts.get(name, 0) + 1

				# determine the recognized face
				name = max(counts, key=counts.get)
			
			# update the list of names
			names.append(name)

		
		# compute fps every 4*batch_size frames
		num_frames = num_frames + 1
		if num_frames%(4*batch_size)==0:
			fps = round((num_frames/(datetime.now().timestamp() \
				- startTime.timestamp())), 2)
			startTime = datetime.now()
			num_frames = 0

		for ((top, right, bottom, left), name) in zip(batch_boxes[k], names):
			# rescale the face coordinates
			top = int(top * r[k])
			right = int(right * r[k])
			bottom = int(bottom * r[k])
			left = int(left * r[k])

			# display bounding box, name and FPS in the image
			cv2.rectangle(frame[k], (left, top), (right, bottom),
				(0, 0, 255), 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(frame[k], name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				0.75, (0, 255, 0), 2)
			cv2.putText(frame[k], "FPS:" + str(fps), 
			    (500,10), 
			    cv2.FONT_HERSHEY_SIMPLEX, 
			    0.5,
			    (100,100,100),
			    2)
		# check to see if we are supposed to display the output frame to
		# the screen
		if args["display"] > 0:
			cv2.imshow("Frame", frame[k])
			key = cv2.waitKey(1) & 0xFF
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
		# if the video writer is None *AND* we are supposed to write
		# the output video to disk initialize the writer
		if writer is None and args["output"] is not None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 35,
				(frame[k].shape[1], frame[k].shape[0]), True)

		# if the writer is not None, write the frame with recognized
		# faces t odisk
		if writer is not None:
			writer.write(frame[k])

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()