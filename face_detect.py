#!/usr/bin/env python3

import cv2
import argparse
import glob
import os
import sys
import json

def detect_faces_haarcascades(image, min_width = 30, min_height = 30):
	# Load the pre-trained face detection model
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

	# Read the image


	# Convert to grayscale (required for face detection)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(min_width, min_height))


	#print(f"Detected {len(faces)} face(s). Output saved as {output_path}")

	if len(faces) > 0:
		#returns (x, y, w, h)
		return faces.tolist()

	return None

def parse_width_height_from_args(arg: str) -> tuple:
    return tuple(map(int, arg.split('x')))

#Filter out faces less than the specified min width and height
def filter_faces_min( faces, min_width, min_height ):
	output_faces = []
	for face in faces:
		if face[2] >= min_width and face[3] >= min_height:
			output_faces.append(face)
	return output_faces
			
#Expands the detected face region by a percentage 0.5 = 50%, 1.0 = 100% (double dimensions)
#faces =  a list of lists of the form [ [ x,y,w,h ], [x,y,w,h] ]
def expand_region(faces, image_width, image_height, percentage_expansion):
	for face in faces:
		x = face[0]
		y = face[1]
		w = face[2]
		h = face[3]

		#Expand the image width by the stated percentage
		new_w = int(w*(1+percentage_expansion))
		#Centre the box horizontally by nudging the X coordinate left by half of the amount we have expanded the box by
		new_x = int(x - (new_w-w)/2)
		#If that takes us off the screen on either the left or right side
		if (new_x < 0) or ((new_w+new_x) > image_width):
			#Calculate the maximum width that would stay on the screen and centred on the bounding box
			
			#Distance from the centre point of the box to the left edge (-1 to cope with rounding)
			x_box_centre = int((x+(w/2)))-1
			max_left = x_box_centre
			max_right = (image_width - x_box_centre)-1
			
			max_width = max_left*2
			if max_right < max_left:
				max_width = max_right*2
			
			new_w = max_width
			new_x = int(x - (new_w-w)/2)
		
		
		#Expand the image width by the stated percentage
		new_h = int(h*(1+percentage_expansion))
		#Centre the box horizontally by nudging the Y coordinate up by half of the amount we have expanded the box by
		new_y = int(y - (new_h-h)/2)
		#If that takes us off the screen on either the left or right side
		if (new_y < 0) or ((new_h+new_y) > image_height):
			#Calculate the maximum height that would stay on the screen and centred on the bounding box
			
			#Distance from the centre point of the box to the left edge (-1 to cope with rounding)
			y_box_centre = int((y+(h/2)))-1
			#Distance from the centre of the box to top of the image
			max_top = y_box_centre
			#Distance from bottom of the image to the centre of the box
			max_bottom = (image_height - y_box_centre)-1
			
			max_height = max_top*2
			if max_bottom < max_top:
				max_height = max_bottom*2
			
			new_h = max_height
			new_y = int(y - (new_h-h)/2)
			
		face[0] = int(new_x)
		face[1] = int(new_y)
		face[2] = int(new_w)
		face[3] = int(new_h)
		
	return faces

#Make sure the face detection has a square aspect ratio
def square_aspect_ratio( faces, image_width, image_height ):
	for face in faces:
		x = face[0]
		y = face[1]
		w = face[2]
		h = face[3]
		
		#Check if already square aspect ratio
		if w == h:
			continue
		
		#Pick the minimum dimension to make square
		square_side = 0
		if w < h:
			square_side = w
			new_h = square_side
			#Nudge the bouding box down by half the difference between the old and new heights to keep the rectangle centred
			new_y = int((h - square_side)/2)+y
			face[1] = new_y
			face[3] = new_h
		else:
			#The height is less than the width
			square_side = h
			new_w = square_side
			new_x = int((w - square_side)/2)+x
			face[0] = new_x
			face[2] = new_w
			
	return faces


def save_image( image, fname ):
	cv2.imwrite(fname, image)
	
def build_dnn_detector_yunet():
	dnn_detector = cv2.FaceDetectorYN.create(
		"face_detection_yunet_2023mar.onnx",
		"",
		(320, 320),
		0.9,		#Score threshold
		0.3,		#NMS threshold
		5000		#Top_k
	)
    
	return dnn_detector
    
def test_dnn_face( dnn_detector, cropped_image , do_scale = True, new_w = 300 ):

	h, w = cropped_image.shape[:2]
	#Don't scale if the image is already smaller than the specified width
	if w < new_w:
		do_scale = False
	aspect = w / h
	new_h = int(new_w / aspect)
	if do_scale:		
		resized = cv2.resize(cropped_image, (new_w, new_h))
		dnn_detector.setInputSize((new_w, new_h))
		faces1 = dnn_detector.detect(resized)
	else:
		dnn_detector.setInputSize((w, h))
		faces1 = dnn_detector.detect(cropped_image)

	face_set = [] 
	if faces1[1] is not None:
		for idx, face in enumerate(faces1[1]):
			x = int(face[0])
			y = int(face[1])
			width = int(face[2])
			height = int(face[3])
			
			#Sometimes produces offscreen coordinates
			if(x < 0):
				x = 0
			if(y < 0):
				y = 0
			
			#Rescale the dimensions back to those of the original image
			if do_scale:
				x = int(x*(w/new_w))
				y = int(y*(h/new_h))
				width = int(width*(w/new_w))
				height = int(height*(h/new_h))
			
			face_set.append( [ x, y, width, height ] )
	return face_set
	
def filter_coords( args, face_coords ):
	if args.expand_percent:
		scaling_percentage = args.expand_percent/100;
		face_coords = expand_region( face_coords, image_width, image_height, scaling_percentage )

	
	if args.square_aspect:
		face_coords = square_aspect_ratio( face_coords, image_width, image_height )

		
	if min_size_set:
		face_coords = filter_faces_min( face_coords,min_width,min_height )

	return face_coords
	
def crop_resize_save(args, x,y,w,h,image,fname):
	if args.highlight:
		#Copies the image otherwise the rectangle stays on the image for the next face, it's a reference
		clone = image.copy()
		cv2.rectangle(clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
		cv2.imwrite(fname, clone)
	else:
		cropped_image = image[y:y+h, x:x+w]
		if args.square_aspect:
			new_image_height, new_image_width = cropped_image.shape[:2]
			assert w == h, "Expected square image"
			assert new_image_width == new_image_height, "Expected square image after save"
		resized = cropped_image
		if args.resize:
			new_width, new_height = args.resize
			resized = cv2.resize(cropped_image, (new_width, new_height))
		save_image( resized, fname )


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--min_size', type=parse_width_height_from_args, help='widthxheight - minimum width and height of a face in pixels e.g. 100x100')
	parser.add_argument('--resize', type=parse_width_height_from_args, help='widthxheight - resize all cropped imaged to this size')
	parser.add_argument('--square_aspect', action="store_true" , help='Always output a square aspect ratio')
	parser.add_argument('--expand_percent', type=float, help='Expand the detected face region by this percentage')
	parser.add_argument('--output_dir', type=str, help='A directory into which to place the cropped versions of the detected faces')
	parser.add_argument('--json', action='store_true', help='Output the detected face coordinates as json to stdout')
	parser.add_argument('--highlight', action='store_true', help='Draw a bounding box on the original image instead of cropping out the face')


	parser.add_argument('files', nargs="*");
	args = parser.parse_args()
	image_file_paths = []
	
	json_output = []
	
	min_size_set = False
	min_width = 30
	min_height = 30
	total_paths = 0
	
	files_inspected = 0
	faces_found = 0
	
	if args.min_size:
		min_size_set = True
		min_width, min_height = args.min_size
	
	#Set the minimum width/height the detector will use as we'll expand the size later
	detect_min_width = min_width
	detect_min_height  = min_height
	if args.expand_percent:
		detect_min_width = int(min_width * (1/(1+(args.expand_percent/100))))
		detect_min_height = int(min_height * (1/(1+(args.expand_percent/100))))
		
	
		
	extensions = ( '*.jpg', '*.png' )
	for dir_path in args.files:
		for ext in extensions:
		    image_file_paths.extend(glob.glob(dir_path+"/"+ext))
		    
	dnn_detector = build_dnn_detector_yunet()
	
	for image_fpath in image_file_paths:
		files_inspected+=1
		#(x, y, w, h)
		image = cv2.imread(image_fpath)
		filename_only  = os.path.basename(image_fpath)
		file_name_without_extension, file_extension = os.path.splitext(filename_only)
		image_height, image_width = image.shape[:2]
		
		
		#Try with the deep neural network at original size
		dnn_face_set = test_dnn_face( dnn_detector, image, False )
		dnn_face_set = filter_coords(args, dnn_face_set)
		
		#If the DNN didn't find anything, progressively scale down the image until it finds something
		#The DNN is only effective when the face is less than about 300 pixels wide
		if len(dnn_face_set) < 1:
			scale_test = int((image_width - 500)/10)
			new_w = int(image_width - scale_test)
			for i in range(0,9):
				dnn_face_set = test_dnn_face( dnn_detector, image, True, new_w )
				dnn_face_set = filter_coords(args, dnn_face_set)
				if len(dnn_face_set) >= 1:
					break
				new_w = int(new_w - scale_test)

		if len(dnn_face_set) >= 1:
			for i, face in enumerate(dnn_face_set):
				x = face[0]
				y = face[1]
				w = face[2]
				h = face[3]
				faces_found+=1
				if(args.output_dir):
					crop_resize_save( args, x,y,w,h, image, args.output_dir+"/"+file_name_without_extension+"_face"+str(i)+file_extension )
			this_obj = { "file" : image_fpath, "face_xywh": dnn_face_set }
			json_output.append( this_obj )
			
		else:
			#If it still didn't find anything then try the Haar Cascade method
			#This false positives a lot, so cut out what it thinks is the face and then check it with the DNN
			face_coords = detect_faces_haarcascades(image, detect_min_width, detect_min_height)
			
			if face_coords is not None and len(face_coords) > 0:
				face_coords = filter_coords(args, face_coords)
				
				if face_coords is not None:
					face_set = []
					for i, face in enumerate(face_coords):
						x = face[0]
						y = face[1]
						w = face[2]
						h = face[3]
						cropped_image = image[y:y+h, x:x+w]
						dnn_face_set = test_dnn_face( dnn_detector, cropped_image, True )
						if len(dnn_face_set) >= 1:
							face_set.append( face )
							faces_found+=1
							if(args.output_dir):
								crop_resize_save( args, x,y,w,h, image, args.output_dir+"/"+file_name_without_extension+"_face"+str(i)+file_extension )
					if len(face_set) > 0:
						this_obj = { "file" : image_fpath, "face_xywh": face_set }
						json_output.append( this_obj )
			else:
				print("No face found in: "+str(image_fpath), file=sys.stderr);
    
	if args.json:
		print(json.dumps(json_output))
	print(f"Files: {files_inspected} Faces Found: {faces_found}", file=sys.stderr)
    
