print('Loading Libraries...')
import pickle
from keras.models import load_model
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for

print('Libraries loaded!!')
print('Loading Variabels and Models..')
app = Flask(__name__)
knn = pickle.load(open('model/knn','rb'))
knn2 = pickle.load(open('model/knn2','rb'))
lightgbm = pickle.load(open('model/lightgbm','rb'))
nn1 = load_model('model/nn.h5')
svc = pickle.load(open('model/svc','rb'))
xgboost = pickle.load(open('model/xgboost','rb'))
lr=pickle.load(open('model/lr','rb'))
catboost= pickle.load(open('model/catboost.sav','rb'))
with open('model/variables/scaler','rb') as f:
    scaler=pickle.load(f)
with open('variables/matchtype_enc','rb')as f:
    matchtype_enc=pickle.load(f)
with open('variables/num_max','rb')as f:
    num_max=pickle.load(f)
with open('variables/win_kill','rb')as f:
    win_kill=pickle.load(f)
print('Variables and Modles are loaded!1')

#CONVERTING MATCH TYPE
def get_prediction(arr):
    temp=np.array(arr[12]).reshape(1,1)
    enc=int(matchtype_enc.transform(temp)[0])
    if(enc==1):
        enc=0.45299
    if(enc==1):
        enc=0.483954
    if(enc==2):
        enc=0.485855
    else:
        enc=0.460818
    arr[12]=enc

    #handling distance
    distance=float(arr[16])+float(arr[18])+float(arr[21])
    arr=np.append(arr,distance)
    arr=np.delete(arr,[16,18,21])

    # hadling win kill
    temp=np.array([int(arr[7]),int(arr[20])]).reshape(1,-1)
    win_k=win_kill.transform(temp)[0][0]
    arr=np.delete(arr,[7,20])
    arr=np.append(arr,win_k)

    # #handling num_max
    temp=np.array([int(arr[12]),int(arr[13])]).reshape(1,-1)
    num_m=num_max.transform(temp)[0][0]
    arr=np.delete(arr,[12,13])
    arr=np.append(arr,num_m)

    print(arr)
    arr=list(map(float,arr))
    arr=np.array(arr)
    arr=arr.reshape(1,20)
    arr=scaler.transform(arr)
    
    pred=list()
    pred.append(nn1.predict(arr)[0][0])
    pred.append(knn.predict(arr)[0])
    pred.append(knn2.predict(arr)[0])
    pred.append(catboost.predict(arr)[0])
    pred.append(lightgbm.predict(arr)[0])
    pred.append(svc.predict(arr)[0])
    pred.append(xgboost.predict(arr)[0])
    pred=np.array(pred).reshape(1,-1)
    final_pred=lr.predict(pred)[0]
    return final_pred

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [str(x) for x in request.form.values()]
    final_features = np.array(int_features)
    prediction = get_prediction(final_features)
    output = round(prediction, 3)
    print('Final predictions', output)
    return render_template('home.html', prediction_text="Win Place Prediction {}".format(output))

@app.route('/predict_api',methods=['POST'])

def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = get_prediction([np.array(list(data.values()))])

    output = prediction
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=False)
