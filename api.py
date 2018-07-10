import flask
from flask import Flask
from flask import request
from flask import render_template
from sklearn.externals import joblib

#below this comment use the modules from machine learning algo
import numpy as np
from scipy import misc


web = Flask(__name__)


@web.route("/")

@web.route("/index")
def index():
   return flask.render_template('index.html')


#app route from the model
@web.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        file = request.files['image']
        if not file:
            return render_template('index.html', label="No file")
        img = misc.imread(file)
        img = img[:,:,:3]
        img = img.reshape(1, -1)
        prediction = model.predict(img)
        label = str(np.squeeze(prediction))
        if label=='10':
            label='0'
        return render_template('index.html', label=label, file=file)


if __name__ == '__main__':
    model = joblib.load('model.pkl')
    web.run(host='0.0.0.0', port=8000, debug=True)
