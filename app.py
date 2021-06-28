import os
from flask import Flask, jsonify, request, render_template
from source.utils import draw_rectangles, read_image, prepare_image
# from config import DETECTION_THRESHOLD
from source.face_detection import detect_faces_with_ssd
from source.face_detection import detect_faces_with_haarcascade
import base64
from source.object_counting import show_original_img
from source.object_counting import medianBlur_unsharp
from source.object_counting import gaussian_grayScale
from source.object_counting import fft
from source.object_counting import find_Canny_edges_Closing
from source.object_counting import contours_count_hatgao
from source.object_counting import contours_count_tools

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/count')
def count():
    return render_template('counting_object.html')


@app.route('/countObject', methods=['POST'])
def count1():
    file = request.files['original_image']

    # Read image
    image = read_image(file)
    img = medianBlur_unsharp(image)

    # encoded_string = base64.b64encode(img.read())
    # Prepare image for html

    to_send1 = prepare_image(img)

    gray = gaussian_grayScale(img)
    to_send2 = prepare_image(gray)

    imgf = fft(gray)
    to_send3 = prepare_image(imgf)

    imgc = find_Canny_edges_Closing(imgf)
    to_send4 = prepare_image(imgc)

    count_image = contours_count_hatgao(image, imgc)
    # count_image = contours_count_tools(image, img)
    to_send5 = prepare_image(count_image)

    return render_template('counting_object.html',
                           image_to_show1=to_send1,
                           image_to_show2=to_send2,
                           image_to_show3=to_send3,
                           image_to_show4=to_send4,
                           image_to_show5=to_send5,
                           init=True)


# ====================================== DETECT ===========================================
@app.route('/detect')
def detect():
    return render_template('detect_faces.html')


# ================== detect using SSD ===========================================
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']

    # Read image
    image = read_image(file)

    # Detect faces
    faces = detect_faces_with_ssd(image)

    # Draw detection rects
    num_faces, image = draw_rectangles(image, faces)

    # Prepare image for html
    to_send = prepare_image(image)

    return render_template('detect_faces.html', face_detected=len(faces) > 0, num_faces=len(faces),
                           image_to_show=to_send,
                           init=True)


# ================== detect using HAAR CASCADE ===========================================
@app.route('/upload_HC', methods=['POST'])
def upload_HC():
    file = request.files['image1']

    # Read image
    image = read_image(file)

    # Detect faces
    faces = detect_faces_with_haarcascade(image)

    # Prepare image for html
    to_send = prepare_image(image)

    return render_template('detect_faces.html', face_detected=len(faces) > 0, num_faces=len(faces),
                           image_to_show=to_send,
                           init=True)


if __name__ == '__main__':
    app.run(debug=True,
            use_reloader=True,
            port=3000)
