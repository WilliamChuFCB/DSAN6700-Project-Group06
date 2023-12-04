import numpy as np
from flask import Flask, request,render_template
import joblib

app = Flask(__name__)
model_file = f"./app/SampleProjectJoblib.joblib"
model = joblib.load(model_file)

mean= [0.58133333,  0.55458333,  0.97783333, 28.93708333,  0.51741667,
        0.09983333,  0.48058333,  0.70075   ,  0.61558333,  0.78858333,
        0.04666667,  0.95683333,  0.09366667,  2.89858333,  3.78408333,
        6.52983333,  0.28383333,  0.50258333,  8.99741667,  4.92491667,
        5.65941667]

std = [ 0.49334054,  0.49701173,  0.14722536,  6.77311781,  0.49969657,
        0.29977765,  0.84159947,  0.45792951,  0.48645708,  0.40831319,
        0.21092389,  0.20323215,  0.29136441,  1.1577124 ,  8.20278184,
       10.62311677,  0.45085693,  0.49999333,  2.89670675,  1.02092074,
        2.16992772]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
 
    string_lst = eval(request.form.get("variable"))
    input_feature = [float(i) for i in string_lst]
    input_feature=[(input_feature[i] - mean[i]) / std[i] for i in range(len(mean))]
    input_feature = np.array(input_feature).reshape(1,21)
    pred = model.predict(input_feature)

    if pred[0] == 0:
        pred = "Low Risk"
    elif pred[0] == 1:
        pred = "High Risk"

    return render_template('index.html', prediction_text='Heart Attack/Disease Risk Assessment: {}'.format(pred))

if __name__ == "__main__":
    app.run(debug=True)
