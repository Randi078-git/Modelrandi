import pandas as pd
import numpy as np
import streamlit as st
import pickle
import datetime
import pytz

def main() :
    st.title ('Sigma Squad Uber Final Presentation')
if __name__ == '__main__' :
    main()

Model = pickle.load(open('modelEN.pkl','rb'))
Scaler = pickle.load(open('scalerEN.pkl','rb'))

#Sidebar
st.sidebar.text('Input Fare Amount Prediction')  
st.sidebar.markdown('Testing')

passenger = st.sidebar.number_input('Passenger', min_value=1, max_value=6, value=1, step=1)
if passenger >1 :
    st.write(f"We have **{passenger}** Passengers")
else :
    st.write(f"We have **{passenger}** Passenger")

distance = st.sidebar.number_input('Distance (km)', min_value=0.1, max_value=500.0, value=4.0, step=0.1)
st.write(f"Radius Distance Total **{distance:.2f}** KM")

Time = st.sidebar.time_input('Pickup Time',value = datetime.datetime.now(pytz.timezone('Asia/Jakarta')), step=60)
today = datetime.date.today()
Date = st.sidebar.date_input('Pickup Date', value=today )


def time_cat(Time):
    hours = Time.hour
    if hours <= 6:
        return 'Early morning'
    elif hours <= 10:
        return 'Morning Rush'
    elif hours <= 16:
        return 'Daytime(Non Rush)'
    elif hours <=19 :
        return 'Evening Rush'
    else :
        return 'Night(Off Peak)'

TimeCategorized = time_cat(Time)

st.write(f"Time Categorized is **{TimeCategorized}**")

def weekend_day(Date):
    dates = Date.weekday()
    if dates>=5:
        return 'Enjoy Your Weekend'
    else :
        return 'Good Luck For Your Weekday'

Daycat = weekend_day(Date)

st.write(f"**{Daycat}**")

def encodeweek(Date):
    dates1 = Date.weekday()
    if dates1>=5:
        return 1
    else :
        return 0

weekend_encoding = encodeweek(Date)

one_hot_columns = ['passenger_count', 'Radius Dist-KM', 'Weekend','Time Category_Daytime(Non Rush)', 
                   'Time Category_Early morning','Time Category_Evening Rush', 'Time Category_Morning Rush',
                    'Time Category_Night(Off Peak)']

new_data = {'passenger_count' : [passenger],
            'Radius Dist-KM' : [distance],
            'Weekend':[weekend_encoding],
            'Time Category' : [TimeCategorized]}
df_newdata = pd.DataFrame(new_data)
#Dfbutton = st.button('Dataframe Model')
#f Dfbutton:
#        st.dataframe(df_newdata)
#with st.expander("Expand Dataframe"):
 #   st.dataframe(df_newdata)

dfonehot = pd.get_dummies(df_newdata)
for kolom in one_hot_columns:
    if kolom not in dfonehot.columns:
            dfonehot[kolom] = 0
#with st.expander("Expand Dataframe"):
 #   st.dataframe(dfonehot)
predictmodel = st.sidebar.button('Calculate Fare')
if predictmodel:
    try:
        df_final_input = dfonehot[one_hot_columns]
        scale = Scaler.transform(df_final_input)
        prediction = Model.predict(scale)
        fare_prediction = prediction[0]
        #st.write(f"Your Fare Amount are $**{fare_prediction:.2f}**")
        st.markdown(f"## Your Fare Amount are $**{fare_prediction:.2f}**")
    except:
        st.write(f"Error please correct the input")




























