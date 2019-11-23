# USAGE
# python3 encode_faces.py --dataset dataset --encodings encodings.pickle

# import the necessary packages
from imutils import paths
from datetime import datetime
import face_recognition
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []
num_of_images_proccessed = 0

# loop over the image paths
startTime = datetime.now()

for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB). Resize image to 1080p so the VRAM doesnt 
	# overflow while trying to pass it through the deep networks

	image = cv2.imread(imagePath)
	image = cv2.resize(image, (1920, 1080))
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# detect the the bounding boxes
	# corresponding to each face in the input image
	
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)
	
	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)
	
# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
print("Total Time Elapsed: "+ str(datetime.now() - startTime))
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()