from flask import Flask, render_template, request, jsonify
import joblib, os
import numpy as np
import pandas as pd

app = Flask(__name__)

BASEDIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASEDIR, 'best_model.pkl'))
age_scaler = joblib.load(os.path.join(BASEDIR, 'age.scaler'))
income_scaler = joblib.load(os.path.join(BASEDIR, 'MonthlyIncome.scaler'))
with open(os.path.join(BASEDIR, 'employee_data.csv'), 'r') as f:
    columns = f.readline().replace('\n','').split(',')[:4]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    try: 
        rq = request.get_json()
        dat = rq.get('features')
        if not dat or len(dat) != 4:
            return jsonify({'error':'잘못된 데이터입니다. '}), 400
        df = pd.DataFrame(np.array([dat]), columns=columns)
        df['Age'] = age_scaler.transform(df[['Age']])
        df['MonthlyIncome'] = income_scaler.transform(df[['MonthlyIncome']])
        pred = model.predict(df)
        prob = model.predict_proba(df)
        return jsonify({
            'prediction': f'{pred[0]}',
            'confidence': f'{float(np.max(prob)) * 100 :.2f}%'
            }), 200
        
    except Exception as e:
        return jsonify({'error':'Internal server error'}), 500
    
if __name__ == '__main__':
    app.run(debug=True)