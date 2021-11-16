import pickle
import streamlit as st
from sklearn import preprocessing 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#loading the trained model
class NeuralNetwork(nn.Module):
  def _init_(self):
    super()._init_()
    self.linear_relu_stack = nn.Sequential(
      nn.Dropout(p=0.5),
      nn.Linear(input_features, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(512, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Linear(512, output_features),
    )

  def forward(self, x):
    logits = self.linear_relu_stack(x)
    return logits

model = open('saved_model','rb')
model = pickle.load(model).to('cpu')

# scaler 
scaler = open('scaler','rb')
min_max_scaler = pickle.load(scaler)

# input features
race_dict = {'white': 1, 'native': 2, 'asian': 3, 'black': 4, 'other': 5}
ethnicity_dict = {'hispanic': 1, 'nonhispanic': 2}
county_dict = {'Hampden County': 1, 'Middlesex County': 2, 'Bristol County': 3, 'Norfolk County': 4, 'Suffolk County': 5, 'Plymouth County': 6, 'Essex County': 7, 'Worcester County': 8, 'Franklin County': 9, 'Hampshire County': 10, 'Barnstable County': 11, 'Berkshire County': 12, 'Dukes County': 13, 'Nantucket County': 14}


@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(query):
    query_scaled = torch.tensor(list(min_max_scaler(query)),dtype=torch.float)
    outputs = model(data)
    outputs = torch.sigmoid(outputs).detach().cpu().numpy()
    outputs = np.array([0 if x<0.5 else 1 for x in outputs])[0]
    if outputs==1:
        return "Adherence"
    return "Non Adherence"
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:white;text-align:center;">Medihouse - Adherence Predictor</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Age = st.number_input("Age")
    Married = st.selectbox('Marital Status',("Married","Single"))
    Race = st.selectbox('Race',('White','Black','Asian','Native','Other'))
    Ethnicity = st.selectbox('Ethnicity',('Hispanic','Non-Hispanic')) 
    Gender = st.selectbox('Gender',("Male","Female"))
    counties = (tuple(county_dict.keys()))
    County = st.selectbox('County',counties)
    
    if Married=="Married":
        Married=0
    else:
        Married=1

    if Gender=='Male':
        Gender=0
    else:
        Gender=1
    
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction([Age,Married,race_dict[Race],ethnicity_dict[Ethnicity],Gender,county_dict[County]]) 
        st.success('{}'.format(result))
     
if __name__=='__main__': 
    main()