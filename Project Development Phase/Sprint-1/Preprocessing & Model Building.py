import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from collections import Counter as c
import pickle


data=pd.read_csv('chronickidneydisease.csv') #dataframe
data.drop(['id'],axis=1,inplace=True)    #dropping id column


data.columns.values[1]="blood_pressure"
data.columns.values[2]="specific_gravity"
data.columns.values[3]="albumin"
data.columns.values[4]="sugar"
data.columns.values[5]="red_blood_cells"
data.columns.values[6]="pus_cell"
data.columns.values[7]="pus_cell_clumps"
data.columns.values[8]="bacteria"
data.columns.values[9]="blood_glucose_random"
data.columns.values[10]="blood_urea"
data.columns.values[11]="serum_creatinine"
data.columns.values[12]="sodium"
data.columns.values[13]="potassium"
data.columns.values[14]="hemoglobin"
data.columns.values[15]="packed_cell_volume"
data.columns.values[16]="white_blood_cell_count"
data.columns.values[17]="red_blood_cell_count"
data.columns.values[18]="hypertension"
data.columns.values[19]="diabetesmellitus"
data.columns.values[20]="coronary_artery_disease"
data.columns.values[21]="appetite"
data.columns.values[22]="pedal_edema"
data.columns.values[23]="anemia"
data.columns.values[24]="class"

#   replacing ckd\t to ckd(chronic kidney disease)
data['class']=data['class'].replace("ckd\t","ckd") 

#   selecting object data dype
catcols=set(data.dtypes[data.dtypes=="O"].index.values)  
catcols.remove('red_blood_cell_count')
catcols.remove('packed_cell_volume')
catcols.remove('white_blood_cell_count')

#    Excluding object data type
contcols=set(data.dtypes[data.dtypes!="O"].index.values)  
contcols.remove('specific_gravity')
contcols.remove('albumin')
contcols.remove('sugar')


contcols.add('red_blood_cell_count')
contcols.add('packed_cell_volume')
contcols.add('white_blood_cell_count')

catcols.add('specific_gravity')
catcols.add('albumin')
catcols.add('sugar')

data['coronary_artery_disease']=data.coronary_artery_disease.replace('\tno','no')
data['diabetesmellitus']=data.diabetesmellitus.replace(to_replace={'\tno':'no','\tyes':'yes','yes':'yes'})

data.packed_cell_volume=pd.to_numeric(data.packed_cell_volume,errors='coerce')
data.white_blood_cell_count=pd.to_numeric(data.white_blood_cell_count,errors='coerce')
data.red_blood_cell_count=pd.to_numeric(data.red_blood_cell_count,errors='coerce')

#  Replacing the missing values by finding mean

data['blood_glucose_random'].fillna(data['blood_glucose_random'].mean(),inplace=True)
data['blood_pressure'].fillna(data['blood_pressure'].mean(),inplace=True)
data['blood_urea'].fillna(data['blood_urea'].mean(),inplace=True)
data['hemoglobin'].fillna(data['hemoglobin'].mean(),inplace=True)
data['packed_cell_volume'].fillna(data['packed_cell_volume'].mean(),inplace=True)
data['potassium'].fillna(data['potassium'].mean(),inplace=True)
data['red_blood_cell_count'].fillna(data['red_blood_cell_count'].mean(),inplace=True)
data['serum_creatinine'].fillna(data['serum_creatinine'].mean(),inplace=True)
data['sodium'].fillna(data['sodium'].mean(),inplace=True)
data['white_blood_cell_count'].fillna(data['white_blood_cell_count'].mean(),inplace=True)

#     Replacing the missing values by finding mode

data['age'].fillna(data['age'].mode()[0],inplace=True)
data['hypertension'].fillna(data['hypertension'].mode()[0],inplace=True)
data['pus_cell_clumps'].fillna(data['pus_cell_clumps'].mode()[0],inplace=True)
data['appetite'].fillna(data['appetite'].mode()[0],inplace=True)
data['albumin'].fillna(data['albumin'].mode()[0],inplace=True)
data['pus_cell'].fillna(data['pus_cell'].mode()[0],inplace=True)
data['red_blood_cells'].fillna(data['red_blood_cells'].mode()[0],inplace=True)
data['coronary_artery_disease'].fillna(data['coronary_artery_disease'].mode()[0],inplace=True)
data['bacteria'].fillna(data['bacteria'].mode()[0],inplace=True)
data['anemia'].fillna(data['anemia'].mode()[0],inplace=True)
data['sugar'].fillna(data['sugar'].mode()[0],inplace=True)
data['diabetesmellitus'].fillna(data['diabetesmellitus'].mode()[0],inplace=True)
data['pedal_edema'].fillna(data['pedal_edema'].mode()[0],inplace=True)
data['specific_gravity'].fillna(data['specific_gravity'].mode()[0],inplace=True)

#  label Encoding
for i in catcols:
    LEi=LabelEncoder()
    data[i]=LEi.fit_transform(data[i])


 #  Spliting Dataset into independent and dependent variables
 
selcols=['blood_urea','blood_glucose_random','coronary_artery_disease','anemia','pus_cell','red_blood_cells','diabetesmellitus',
                   'pedal_edema']
x=pd.DataFrame(data,columns=selcols)
y=pd.DataFrame(data,columns=['class'])


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

lgr=LogisticRegression()
lgr.fit(x_train,y_train)

y_pred=lgr.predict(x_test)
y_pred1=lgr.predict([[129,99,1,0,0,1,0,1]])


accur=accuracy_score(y_test,y_pred)

conf=confusion_matrix(y_test,y_pred)
print(conf)

model=pickle.dump(lgr,open('CKD.pkl','wb'))