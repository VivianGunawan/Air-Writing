# USAGE Instructions
# python tracker.py --video yourfile.mp4
# python tracker.py


# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of your sticker  in the HSV color space
lower_threshold = None
upper_threshold = None
pts = deque(maxlen=args["buffer"])
tracked = []

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	#Use the function cv2.flip to have a mirror image on the camera feed so you don't have to write your number backwards
	frame= pass
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	# convert blurred into the HSV color space using cv2.cvtColoor function
	hsv = pass

	# construct a mask for the your sticker, then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = pass
	
	# find contours in the mask using cv2.findContours function
	cnts = pass
	cnts = imutils.grab_contours(cnts)


	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask
		# Use the function cv2.minEnclosingCircle and assign it to ((x,y),radius)
		# centroid
		pass
                # only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame, using the function cv2.circl
			pass
	# update the points queue
	pts.appendleft(center)
	

	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

        #set the charcater "a" to be a pen down function
	if key==ord("a"):
		tracked.append(center)
        #set the charcater "s" to be a pen up function or end of a stroke
	elif key==ord("s"):
		tracked.append(None)
	# if the 'q' key is pressed, stop the loop
	elif key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
