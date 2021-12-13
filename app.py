from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = Flask(__name__)
model = load_model('model.h5')
scaler = MinMaxScaler(feature_range=(0,1))

@app.route('/')
def index():
    return render_template('index.html', predict_close=0)

@app.route('/predict', methods=['POST'])
def predict():
    #Preprocessing
    close = [x for x in request.form.values()]
    temp_hasil = scaler.fit_transform(np.array(close).reshape(-1,1))
    #Prediksi
    prediction = model.predict(temp_hasil)
    hasil_predict = scaler.inverse_transform(prediction)
    hasil_predict = hasil_predict[0,0]

    return render_template('index.html', predict_close=str(hasil_predict).replace('.',','))


if __name__ == '__main__':
    app.run(debug=True)