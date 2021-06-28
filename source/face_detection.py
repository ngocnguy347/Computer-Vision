import cv2 # OpenCV for image editing, computer vision and deep learning
import numpy as np # Numpy for math/array operations
from .utils import get_folder_dir # Custom function for better directory name handling


def detect_faces_with_ssd(image, min_confidence = 0.2):
    '''Detect face in an image'''
    
    faces_list = []
    
    # Get models directory
    models_dir = get_folder_dir("models") 
    # Assign prototxt and model filenames
    prototxt_filename = "deploy.prototxt.txt"
    model_filename = "res10_300x300_ssd_iter_140000.caffemodel"
    # Load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(models_dir + prototxt_filename, 
                                   models_dir + model_filename)
    
    # Get image shape
    (image_height, image_width) = image.shape[:2]
    # Construct an input blob for the image 
    # by resizing to a fixed 300x300 pixels and then normalizing it
    resized_image = cv2.resize(image, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_image,
                                 scalefactor=1.0,
                                 size=(300, 300),
                                 mean=(104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detected_faces = net.forward()
    
    # Calculate number of detections
    num_detected_faces = detected_faces.shape[2]
    
    # Loop over the detections
    for index in range(0, num_detected_faces):
        # Initialize a dictionary that will contain detection details
        face_dict = {}
        
        # Extract the confidence (i.e., probability) associated with the
        # prediction 
        confidence = detected_faces[0, 0, index, 2]
        # Convert it to a native python variable (float)
        confidence = confidence.item()
        # Filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > min_confidence:
            # Get detection coords
            rect = detected_faces[0, 0, index, 3:7] * np.array([image_width, image_height, image_width, image_height])
            # Reformat detection coords
            (start_x, start_y, end_x, end_y) = rect.astype("int")
            # Convert them to native python variables (int)
            start_x = start_x.item()
            start_y = start_y.item()
            end_x = end_x.item()
            end_y = end_y.item()
            # Ensure coords are btw [0, image size]
            # start_x = max(0, start_x)
            # start_y = max(0, start_y)
            # end_x = min(end_x, image_width)
            # end_y = min(end_y, image_height)
            
            # Add detection coords to dictionary
            face_dict['rect'] = (start_x, start_y, end_x, end_y)
            
            # Add detection confidence (probability) to dictionary
            # face_dict['prob'] = confidence * 0

            faces_list.append(face_dict)
            
    # Return the face image area and the face rectangle
    return faces_list


def detect_faces_with_haarcascade(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces_list = []
    # Read the input image

    # img = cv2.imread('/content/gdrive/MyDrive/Assets/detect_recognition_Object/people4.jpg')

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # các tham số của detectMultiScale(img, scaleFactor, minNeighbors)
    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    # Draw rectangle around the faces
    # x, y - vị trí của góc trên cùng bên trái của hình chữ nhật
    # w, h - chiều rộng và chiều cao của hình chữ nhật
    # (255, 0, 0): mã màu RGB, tiếp sau đó là độ dày đường viền = 2

    for (x, y, w, h) in faces:
        a = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        faces_list.append(a)

    # Display the output
    return faces_list
