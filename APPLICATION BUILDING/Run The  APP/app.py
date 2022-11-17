                                #  Chronic Kidney Disease Prediction....
from flask import Flask,render_template,request
import pandas as pd
import numpy as np            #importing necessary modules...
import pickle


app=Flask(__name__)      #initializing a flask app
model=pickle.load(open('CKD.pkl','rb'))   #loading model

@app.route('/')
def home():
    return render_template('index.html')  #rendering a index page


@app.route('/home',methods=['POST','GET'])
def index():
    return render_template('home.html') #rendering an home page

@app.route('/predict',methods=['POST','GET'])
def predict():
    return render_template('predict.html')  #rendering a prediction page

@app.route('/backindex',methods=['POST','GET']) 
def backindex():
    return render_template('index.html')     #rendering back to the indexpage

@app.route('/backhome',methods=['POST','GET'])
def backhome():
    return render_template('home.html')      #rendering back to the homepage

@app.route('/result',methods=['POST'])   #rendering prediction button
def result():
    input_features=[float(x) for x in request.form.values()]
    features_value=[np.array(input_features)]
    features_name=['blood_urea','blood_glucose_random','coronary_artery_disease','anemia','pus_cell','red_blood_cells','diabetesmellitus',
                   'pedal_edema']
    
    df=pd.DataFrame(features_value,columns=features_name)
    output=model.predict(df)   #predicting with the help of model
    b=[1]
    if output==b:
        return render_template("result1.html")  
    else:
        return render_template("result2.html")

    
    
if __name__=="__main__":      # Running our app
    app.run(debug=True)
    

