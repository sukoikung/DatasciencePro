from django.shortcuts import render
import numpy as np
from joblib import load

from .models import *
# Create your views here.

def index(req):
    result = 'null'
    data = 'null'
    Accuracy = 'null'
    if req.method == 'POST':
        print('เขา POST มา')
        #print(req.POST)
        data = str(req.POST['data'])


        train = load('./myapp/static/train.model')
        model = load('./myapp/static/chatgroup.model')
        
        def predict_category(s, train=train, model=model):
            pred = model.predict([s])
            print(pred[0])
            return train.target_names[pred[0]]

        result = predict_category(data)
        Accuracy = round((model.score(train.data, train.target)*100),2)

    else:
        print('เขากด enter มา')
    return render(req, 'myapp/index.html', { 
        'result': result,
        'Accuracy':Accuracy
    })
