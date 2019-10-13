from flask import Flask,render_template,jsonify,request
import pandas as pd
import numpy as np
import pickle
app=Flask(__name__)
model=pickle.load(open('model.pk1','rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('home.html', prediction_text='The disease is {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
