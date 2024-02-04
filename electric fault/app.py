import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
 
app = Flask(__name__)

model = tf.keras.models.load_model('electricfault.h5',compile=False)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()] 
    features = [np.array(features)]  
    u=np.array([ 1.37211943e+01, -4.48452685e+01,  3.43923943e+01, -7.66709002e-03,
        1.15210285e-03,  6.51498717e-03])
    std=np.array([4.64712110e+02, 4.39241255e+02, 3.71083807e+02, 2.89131799e-01,
       3.13417112e-01, 3.07877546e-01])
    features=(features-u)/std
    features =features.reshape(1,6 , 1)
    a=model.predict(features)
    for i in range(0,1):
      for j in range(4):
       if a[i][j]>=0.5:
          a[i][j]=1
       else:
          a[i][j]=0


    if (np.all(a[0] == [0,0,0,0])):
       b="There is no fault in the power line"
    elif (np.all(a[0] == [1,0,0,1])):
       b="There is a LG Fault (Between Phase A and Gnd)"
    elif (np.all(a[0] == [0,0,1,1])):
       b="There is a LL Fault (Between Phase A and Phase B)"
    elif (np.all(a[0] == [1,0,1,1])):
       b="There is a LLG Fault (Between Phase A,B and ground)"
    elif (np.all(a[0] == [0,1,1,1])):
       b="There is a LLL Fault (Between all three phases)"
    elif (np.all(a[0] == [1,1,1,1])):
       b="There is a LLLG Fault (Three phase symmetrical fault)"

    

    return render_template('index.html', prediction_text=f'{b}')



if __name__ == "__main__":
    app.run()