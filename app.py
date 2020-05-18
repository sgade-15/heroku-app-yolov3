from flask import Flask, render_template, request
import numpy
import cv2
import os
import base64

#LEAF_FOLDER = os.path.join('static', 'leaf_photo')

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = LEAF_FOLDER

from Inference import get_plant_disease, background_removal, object_detection

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        if 'file' not in request.files:
            print("files not uploaded")
            return
        #file = request.files['file']
        #read image file string data
        filestr = request.files['file'].read()
        #convert string data to numpy array
        npimg = numpy.fromstring(filestr, numpy.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        img = object_detection(image_bytes=img)
        #cv2.imwrite('static/leaf_image.png',img)
        retval, buffer_img= cv2.imencode('.jpg', img)
        data = base64.b64encode(buffer_img).decode('utf-8')
        dim = (256, 256)
        # resize image
        image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	#leaf_image = os.path.join(app.config['UPLOAD_FOLDER'], 'leaf_image.png')
        foreground_image = background_removal(image_bytes=image)
        top1_prob, disease_name, top3_disease, top3_prob = get_plant_disease(image_bytes=foreground_image)
        return render_template('result2.html', leaf_image=data ,disease=disease_name, probability=top1_prob, top3= top3_disease, top3_prob= top3_prob)


if __name__ == '__main__':
    app.run(debug=True,port=os.getenv('PORT',5000))
