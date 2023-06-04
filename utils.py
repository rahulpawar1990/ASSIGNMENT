import pickle
import numpy as np
import pandas as pd
import json
import config 
 





class MedicalInsurance():
    def __init__(self,age,sex,bmi,children,smoker,region):
        self.age=age
        self.sex=sex
        self.bmi=bmi
        self.smoker=smoker
        self.children=children
        
        self.region='region_'+region
    def load_models(self):
        with open(config.MODEL_FILE_PATH,'rb') as f:
            self.model=pickle.load(f)

        with open(config.JSON_FILE_PATH,'r') as f:
            self.json_data=json.load(f)
    def get_predicted_charges(self):
        self.load_models()

        region_index=region_index=list(self.json_data['columns']).index(self.region)

        test_array=np.zeros(len(self.json_data['columns']))
        test_array[0]=self.age
        test_array[1]=self.json_data['sex'][self.sex]
        test_array[2]=self.bmi
        test_array[3]=self.children
        test_array[4]=self.json_data['smoker'][self.smoker]
        test_array[region_index]=1

        print('test_array \n',test_array)
        charges=round(self.model.predict([test_array])[0],2)
        return charges
    
if __name__== '__main__':
    age=19
    sex='male'
    bmi=27
    children=1
    smoker='yes'
    region='northeast'
    
    med_ins=MedicalInsurance(age,sex,bmi,children,smoker,region)
    primium=med_ins.get_predicted_charges()
    print('predicted charges :',primium ,'\ Rs.')
