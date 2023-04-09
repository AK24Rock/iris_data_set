from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
# load the model
model = pickle.load(open("savedmodel.sav", 'rb'))


@app.route('/')
def home():
    result = ''
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    SepalLengthCm = float(request.form['sepal_length'])
    SepalWidthCm = float(request.form['sepal_width'])
    PetalLengthCm = float(request.form['petal_length'])
    PetalWidthCm = float(request.form['petal_width'])
    result = model.predict(
        [[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])[0]
    return render_template('result.html', **locals())


if __name__ == '__main__':
    app.run(debug=True)
