from flask import Flask, templating
from markupsafe import escape
from flask import request 
from flask import render_template
from keras.models import load_model
import os
import numpy as np
from PIL import Image,ImageOps
from werkzeug.utils import redirect

app = Flask(__name__,template_folder='templates')

cell_dict = {0:"Benign", 1:"Malignant", 2:"Normal"}
model = load_model('lung_cancer_prediction.h5')

def predict(fname):
    img = Image.open(fname)
    img = ImageOps.grayscale(img)
    img = img.resize((64,64))
    img = np.reshape(img,[-1,64,64,1])
    return cell_dict[np.argmax(model.predict(img))]


@app.route('/')
def index():
    return render_template('upload2.html')
'''
@app.route('/success',methods = ['POST'])
def upload():
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)
        cname = predict(f)  
        return render_template("result.html", name = cname,user_image = f.filename)  
    else:
        return "jojojojo"
'''  

@app.route('/success2',methods = ['POST'])
def upload2():
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)
        cname = predict(f)  
        return render_template("result.html", name = cname,user_image = f)
    else:
        return "jojojojo"

