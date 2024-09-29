from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)

# Path to the Haar Cascade XML file for car detection
car_cascade_src = 'cars.xml'  # Ensure this file is in the same directory as app.py
car_cascade = cv2.CascadeClassifier(car_cascade_src)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file
    image_path = os.path.join('Data', file.filename)
    file.save(image_path)

    # Process the image for car detection
    img_car = cv2.imread(image_path)
    img_car_gray = cv2.cvtColor(img_car, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_car_gray, (5, 5), 0)
    dilated = cv2.dilate(blur, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    cars = car_cascade.detectMultiScale(closing, 1.1, 1)

    # Creating the bounding boxes
    cnt = 0
    for (x, y, w, h) in cars:
        cv2.rectangle(img_car, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cnt += 1

    # Save the detected image
    output_image_path = os.path.join('static', 'detected_' + file.filename)
    cv2.imwrite(output_image_path, img_car)

    return f'Number of detected cars: {cnt}<br><img src="{url_for("static", filename="detected_" + file.filename)}" alt="Detected Cars"/>'


if __name__ == '__main__':
    if not os.path.exists('Data'):
        os.makedirs('Data')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
