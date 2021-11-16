 
import pickle
import streamlit as st
from sklearn import preprocessing 
#loading the trained model


# scaler 
scaler = open('scaler','rb')
min_max_scaler = pickle.load(scaler)

# input features
input_features = open('input_features','rb')
input_features = pickle.load(input_features)
print(input_features)

@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction():
    return "Adherence"
  
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
    Gender = st.selectbox('Gender',("Male","Female"))
    Married = st.selectbox('Marital Status',("Unmarried","Married")) 
    County = st.number_input("County") 
    Age = st.number_input("Age")
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction() 
        st.success('{}'.format(result))
     
if __name__=='__main__': 
    main()