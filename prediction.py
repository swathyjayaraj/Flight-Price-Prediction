import streamlit as st 
import pandas as pd
import numpy as np
import pickle
import requests
from pycaret.regression import load_model, predict_model, setup
from datetime import date
import datetime
from PIL import Image
import pandas_profiling as pf
from pandas_profiling import ProfileReport


model = load_model('model')


def predict(model, input_df):
    setup(data=input_df, target='Label')
    
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions


#Data = pickle.load(open("model.pkl","rb"))

print("success")


# navigation bar
#st.sidebar.image("images/compunnel.png",width=100)
st.sidebar.title('Flight price Prediction')
page = st.sidebar.radio(
    "What would you like to know?",
    ('Data Analysis','Predict the Price', )
)




#Data Analysis
if page == 'Data Analysis':
    # Open image 
    
    ## Introduction
    st.header('FLIGHT FARE PREDICTION PROJECT')
    st.markdown("#### Introduction")
    st.markdown( "The price of airline tickets can be influenced by numerous factors, such as the airline, travel dates, departure and arrival locations, flight path, duration, and more. The issue of fluctuating flight prices is compounded by travelers' desire to obtain air tickets at the lowest possible price. As a result, the airfare pricing strategy has evolved into a complex system of sophisticated rules and mathematical models." )
    image1 = Image.open('E:/UEL/Thesis/New folder/Aircraft Pics/airimg.jpg')
    st.image(image1)
    st.markdown("#### Factors influencing the Price!")
    st.markdown("There are several factors that affect the price of plane tickets. Below are a few of them. Please select the parameter you are interested in learning more about from the list.")
    choose = st.radio("What factor would you like to explore?",
    ('Airline','Duration', 'Total Stops', 'Source', 'Destination', 'Journey Date'))
    if choose == 'Airline':
        st.markdown("#### Airline")
        st.markdown('A total of 12 airlines are listed in the dataset. A Jet Airways flight is the most frequent between the cities. Indigo operates the second-highest number of flights. Air India and Spicejet are the other two major airlines in this dataset. We could see that number of premium flights is comparatively lesser than the economical ones thus referring to the purchasing power of the population. ')
        image3 = Image.open('E:/UEL/Thesis/New folder/Aircraft Pics/Airline.PNG')
        st.image(image3)

    if choose == 'Duration':
        st.markdown("#### Duration")
        st.markdown('The duration of a flight refers to the length of time it takes to travel from the departure airport to the destination airport. This can vary depending on factors such as distance, weather conditions, and flight route. The price of a flight can vary depending on various factors such as time of booking, demand for the route, and airline policies. In general, longer flights tend to be more expensive than shorter flights, but there are exceptions to this depending on the route and other factors. ')
        st.markdown('From the below scatter plot there is no clear pattern, as there should have been, regarding the effect of flight duration (or distance) on air ticket prices. We could see a few outlier points in the graph, and this corresponds to other factors like premium airlines and the date of the flight (higher if it occurs on a public holiday).')
        image4 = Image.open('E:/UEL/Thesis/New folder/Aircraft Pics/Duration.PNG')
        st.image(image4)

    if choose == 'Total Stops':
        st.markdown("#### Total Stops")
        st.markdown('A flights price is closely related to its total number of stops. The cost of nonstop flights is generally higher than that of flights with one or more stops. Due to the shorter travel time and comfort of not having to switch planes, non-stop flights are more convenient for passengers willing to pay a premium. However, we see the opposite trend in our dataset. Since the average fare is usually the lowest when compared to flights with multiple stops or layovers, we are essentially paying for one flight for the entire trip. During non-stop flights, fuel is consumed more efficiently, crew and aircraft are utilized better, and ground services are used more efficiently, which reduces airline costs per seat mile.')
        image5 = Image.open('E:/UEL/Thesis/New folder/Aircraft Pics/Total_stops.PNG')
        st.image(image5)

    if choose == 'Source':
        st.markdown("#### Source")
        st.markdown("The below plot shows the average airfare from five major Indian cities.It is evident that Bangalore and Delhi have a price that is higher than the mean. There were also a few outliers in both cities, which can be attributed to the Premium Jet services from each city. This may be due to Bangalore's status as the Silicon Valley of India, where the majority of people work in the IT industry and travel home during the holidays. Like Delhi, which is India's capital city, it is a city with a smaller native population and a larger urban population that commutes there for work. This can be further supported by the fact that Bangalore and Delhi have the highest number of IT companies and multinational companies that employ professionals from all over India.")
        image6 = Image.open('E:/UEL/Thesis/New folder/Aircraft Pics/Source.PNG')
        st.image(image6)

    if choose == 'Destination':
        st.markdown("#### Destination")
        st.markdown("Like the source, the destination can also affect the price of a flight. Vacation spots and major cities may be more expensive than smaller, less popular cities.Several factors may contribute to the high airfare prices in Delhi and New Delhi. As of 2018, jet fuel prices in Delhi increased by 26.4%. Delhi is also the National Capital and seat of government, as well as a popular tourist destination, which could result in higher flight prices. In addition, Bangalore and Cochin are only two of the major airports in south India, a major tourist destination and home to many people in the country.")
        image7 = Image.open('E:/UEL/Thesis/New folder/Aircraft Pics/Destination.PNG')
        st.image(image7)

    if choose == 'Journey Date':
        st.markdown("#### Journey Date")
        st.markdown("To determine the trend, we extracted months from the Journey date and plotted them against the price. The data shows that the total count of flights is highest in May, and the sum of fare is also maximum during this month. This could be due to summer vacations for schools and colleges, when families tend to travel more. April, on the other hand, has the lowest number of flights, likely due to final exams for schools and colleges. In addition, due to the end of the first quarter, offices are usually busier during April, resulting in a decrease in flight demand. Using this information, we can see how seasonality affects flight availability and pricing.")
        image8 = Image.open('E:/UEL/Thesis/New folder/Aircraft Pics/Journey.PNG')
        st.image(image8)
    


if page == 'Predict the Price': 
    st.header('Welcome to the Flight Fare Prediction!')
    image2 = Image.open('E:/UEL/Thesis/New folder/Aircraft Pics/predict.jpg')
    st.image(image2)
    st.markdown("Select from the below attributes to get an estimate of your flight price.")
    Airline=st.selectbox('Airline: The name of the airline', ['Jet Airways','IndiGo','Air India','Multiple carriers','SpiceJet','Vistara','Air Asia','GoAir','Multiple carriers Premium economy','Jet Airways Business','Vistara Premium economy','Trujet'])
    if Airline == "Jet Airways":
        air_inp = 4
    elif Airline == "IndiGo":
        air_inp = 3
    elif Airline == "Air India":
        air_inp = 1
    elif Airline == "Multiple carriers":
        air_inp = 5
    elif Airline == "SpiceJet":
        air_inp = 6
    elif Airline == "Vistara":
        air_inp = 7
    elif Airline == "Air Asia":
        air_inp = 0
    elif Airline == "GoAir":
        air_inp = 2

    #source
    Source=st.selectbox('Source: The source from which the service begins', ['Delhi','Kolkata','Banglore','Mumbai','Chennai'])
    if Source == "Bangalore":
        source_inp = 0
    elif Source == "Chennai":
        source_inp = 1
    elif Source == "Delhi":
        source_inp = 2
    elif Source == "Kolkata":
        source_inp = 3
    elif Source == "Mumbai":
        source_inp = 4

    #Destination
    dest_inp=0
    Destination=st.selectbox(' Destination:', ['Banglore','Cochin','Delhi','New Delhi','Hyderabad','Kolkata'])
    if Destination == "Bangalore":
        dest_inp = 0
    elif Destination == "Cochin":
        dest_inp = 1
    elif Destination == "Delhi":
        dest_inp = 2
    elif Destination == "New Delhi":
        dest_inp = 3
    elif Destination == "Hyderabad":
        dest_inp = 4
    elif Destination == "Kolkata":
        dest_inp = 5


    #Number of Stops
    Total_Stops = st.selectbox('Total_Stops: Total stops between the source and destination', ['1 stop', '2 stops','3 stops','4 stops','non-stop'])
    if Total_Stops=='non-stop':
        stop_inp=0
    elif Total_Stops=='1 stop':
        stop_inp=1
    elif Total_Stops=='2 stops':
        stop_inp=2
    elif Total_Stops=='3 stops':
        stop_inp=3
    elif Total_Stops=='4 stops':
        stop_inp=4

    #Date of Journey
    Date_of_Journey = st.date_input("Date of journey",datetime.datetime.now())
    Journey_Date=int(pd.to_datetime(Date_of_Journey, format="%Y/%m/%d").day)
    Journey_Month=int(pd.to_datetime(Date_of_Journey, format = "%Y/%m/%d").month)

    #Departure time
    Dep_Time = st.time_input("Departure Time",datetime.time())
    str1=str(Dep_Time)
    list1=str1.split(':')
    Dep_Hour=int(list1[0])
    Dep_Min=int(list1[1])

    #Arrival time
    Arrival_Time=st.time_input("Arrival Time",datetime.time())
    str2=str(Arrival_Time)
    list2=str2.split(':')
    Arr_Hour=int(list2[0])
    Arr_Min=int(list2[1])

    #Duration
    Duration=abs((Arr_Hour*60 +Arr_Min*1)-(Dep_Hour*60+Dep_Min*1))

    rfr_model = pickle.load(open("model.pkl","rb"))
    par = [Duration,stop_inp,Journey_Date, Journey_Month,Dep_Hour, Dep_Min, Arr_Hour, Arr_Min, air_inp,source_inp ,dest_inp]
    arrays=np.array(par,dtype="int64")
    
    if st.button("PREDICT"):
        pred = rfr_model.predict([arrays])[0]
        st.write("Your Fare Price is : " , round(pred ,3)  , "INR")









