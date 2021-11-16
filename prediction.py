import pickle
import streamlit as st
from sklearn import preprocessing 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,random_split ,DataLoader


#loading the trained model
class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = nn.Sequential(
      nn.Dropout(p=0.5),
      nn.Linear(6, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(512, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Linear(512, 1),
    )

  def forward(self, x):
    logits = self.linear_relu_stack(x)
    return logits
class CSVDataset(Dataset):
  
  # load the dataset
  def __init__(self, x,y):
    # store the inputs and outputs
    self.X = x
    self.y = y
 
  # number of rows in the dataset
  def __len__(self):
    return len(self.X)
 
  # get a row at an index
  def __getitem__(self, idx):
    x_data=torch.tensor(self.X.iloc[idx].values,dtype=torch.float)
    if len(self.y)==0:
       return [x_data,[]] 
    y_data=torch.tensor(int(self.y.iloc[idx]),dtype=torch.float)
    return [x_data,y_data]

device = torch.device('cpu')
model = NeuralNetwork()
model.load_state_dict(torch.load("model_best.pth", map_location=device)['model_state_dict'])


# scaler 
scaler = open('scaler','rb')
min_max_scaler = pickle.load(scaler)

# input features
race_dict = {'White': 1, 'Native': 2, 'Asian': 3, 'Black': 4, 'Other': 5}
ethnicity_dict = {'Hispanic': 1, 'Non-hispanic': 2}
county_dict = {'Hampden County': 1, 'Middlesex County': 2, 'Bristol County': 3, 'Norfolk County': 4, 'Suffolk County': 5, 'Plymouth County': 6, 'Essex County': 7, 'Worcester County': 8, 'Franklin County': 9, 'Hampshire County': 10, 'Barnstable County': 11, 'Berkshire County': 12, 'Dukes County': 13, 'Nantucket County': 14}


@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def predict(query):
    
    data_dict={}
    data_dict["AGE"]=[query[0]]
    data_dict["MARITAL"]=[query[1]]
    data_dict["RACE"]=[query[2]]
    data_dict["ETHNICITY"]=[query[3]]
    data_dict["GENDER"]=[query[4]]
    data_dict["COUNTY"]=[query[5]]
    query_df = pd.DataFrame(data_dict)
    query_scaled = list(min_max_scaler.transform(query_df))
    model.eval()
    query_scaled = torch.tensor(query_scaled,dtype=torch.float)
    outputs = model(query_scaled)
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
    Race = st.selectbox('Race',tuple(race_dict.keys()))
    Ethnicity = st.selectbox('Ethnicity',(ethnicity_dict.keys())) 
    Gender = st.selectbox('Gender',("Male","Female"))
    County = st.selectbox('County',tuple(county_dict.keys()))
    
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
        result = predict([Age,Married,race_dict[Race],ethnicity_dict[Ethnicity],Gender,county_dict[County]]) 
        st.success('{}'.format(result))
     
if __name__=='__main__': 
    main()