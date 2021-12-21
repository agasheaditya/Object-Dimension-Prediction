# USAGE
# python measureSize-withobject.py --image .\inputs\kicker6.jpg --width 8 --object "kicker"

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

""" Constants """
# key :  (lenght, width)
#consts = {"wall_panel":(2050,600), "kicker":(1600, 130), "ledger_pipe-1":(1160,0), "ledger_pipe-2":(850,0), "ledger_pipe-3":(1750,0),
#		  "verticle_pipe-1":(1080,0), "verticle_pipe-2":(2080,0), "U_jack":(680,), "U_jack-with_nut":(870,0),
#		  "base_jack-1":(760,0), "base_jack-2":(880,0)}

wall_panel = {"wall_panel":(2050,600)}
kicker = {"kicker":(1600, 130)}
ledger_pipe = {"lp-1":1160, "lp-2":1750}
vertical_pipe = {"vp-1":1080,"vp-2":2080}
U_jack = {"U_jack":680, "U_jack-with_nut":870}
base_jack = {"bj-1":760, "bj-2":870}



def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
#ap.add_argument("-w", "--width", type=float, required=True,
#	help="width of the left-most object in the image (in inches)")
ap.add_argument("-o", "--object", required=True)
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
#cv2.imshow("original", image)

scale_percent = 20 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9,9), 0)
#gray = cv2.bilateralFilter(image ,9,75,75)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# loop over the contours individually
counter = 1
record = []
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 100:
		continue

	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")


	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)

	if pixelsPerMetric is None:
		pixelsPerMetric = dB / 8#args["width"]

	# compute the size of the object
	dimA = (dA / pixelsPerMetric) * 2.54
	dimB = (dB / pixelsPerMetric) * 2.54
	#print(dimA,"x", dimB)

	# draw the object sizes on the image
	cv2.putText(orig, "{:.1f}cm".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}cm".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

	if counter ==2:
		record.append([dimA, dimB])# length , width
		cv2.imwrite("op_0.jpg", orig)
	counter +=1

	# show the output image
	cv2.imshow("Image", orig)

	cv2.waitKey(1)

print("\n--- \nsaved records: \nLength:",record[0][0],"Width:", record[0][1])
#print(args["object"])
#temp = args["object"]

if args["object"] == "wall_panel":
	for key, val in wall_panel.items():
		#print(key, val, int(val[0]*0.8) , int(val[0]*1.2), record[0][0])
		if int(record[0][0]*10) in range(int(val[0]*0.9) , int(val[0]*1.1)) and int(record[0][1]*10) in range(int(val[1]*0.9) , int(val[1]*1.1)):
			print("Found:",key)
elif args["object"] == "kicker":
	for key, val in kicker.items():
		if int(record[0][0] * 10) in range(int(val[0] * 0.9), int(val[0] * 1.1)) and int(record[0][1]*10) in range(int(val[1]*0.9) , int(val[1]*1.1)):
			print("Found:", key)
elif args["object"] == "ledger_pipe":
	for key, val in ledger_pipe.items():
		if int(record[0][0] * 10) in range(int(val * 0.75), int(val * 1.25)):
			print("Found:", key)
elif args["object"] == "vertical_pipe":
	for key, val in vertical_pipe.items():
		if int(record[0][0] * 10) in range(int(val * 0.75), int(val * 1.25)):
			print("Found:", key)
elif args["object"] == "U_jack":
	for key, val in U_jack.items():
		if int(record[0][0] * 10) in range(int(val * 0.9), int(val * 1.1)):
			print("Found:", key)
elif args["object"] == "base_jack":
	for key,val in base_jack.items():
		if int(record[0][0] * 10) in range(int(val * 0.95), int(val * 1.05)):
			print("Found:", key)


#for key, val in args["object"].items():
#	pass