#importing the libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

df= pd.read_excel('E:\\Excercises\\EDA projects\\Flight Prediction\\Data_Train.xlsx')

#there are 2 nan values so dropping them off
df.dropna(inplace=True)
#Dropping the coloumns as routes is similar to Total_stops and Addn info is 80% empty
df.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

#Date oj journey should be splitted into date month and year 
df['Journey_Date']=df['Date_of_Journey'].str.split('/').str[0]
df['Journey_Month']=df['Date_of_Journey'].str.split('/').str[1]
df['Journey_Date']=df['Journey_Date'].astype(int)
df['Journey_Month']=df['Journey_Month'].astype(int)
df.drop('Date_of_Journey',axis=1,inplace=True)

df['Dep_Hour']=df['Dep_Time'].str.split(':').str[0]
df['Dep_Min']=df['Dep_Time'].str.split(':').str[1]
df['Dep_Hour']=df['Dep_Hour'].astype(int)
df['Dep_Min']=df['Dep_Min'].astype(int)
df.drop('Dep_Time',axis=1,inplace=True)

# Arrival time feature engineering. We need to split the time and date
df['Arrival_Time'].str.split(' ')
df['Arrival_Time']=df['Arrival_Time'].str.split(' ').str[0]
df['Arr_Hour']=df['Arrival_Time'].str.split(':').str[0]
df['Arr_Min']=df['Arrival_Time'].str.split(':').str[1]
df['Arr_Hour']=df['Arr_Hour'].astype(int)
df['Arr_Min']=df['Arr_Min'].astype(int)
df.drop('Arrival_Time',axis=1,inplace=True)

df['Duration']=  df['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)

#Dropping the coloumns as routes is similar to Total_stops and Addn info is 80% empty
#df.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

df['Total_Stops'].replace(['1 stop', 'non-stop', '2 stops', '3 stops', '4 stops'], [1, 0, 2, 3, 4], inplace=True)

nominal_data = df[['Airline','Source','Destination']]
df_nominal=pd.DataFrame(nominal_data)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_nominal = df_nominal.apply(LabelEncoder().fit_transform)
df= df.drop(['Airline', 'Source','Destination'], axis=1)
df = pd.concat([df, df_nominal], axis=1)

# Move column 'Price' to the last position
col_to_move = 'Price'
last_position = len(df.columns) - 1

col = df.pop(col_to_move)
df.insert(last_position, col_to_move, col)
list(df.columns.values)
df_col=list(df)


from sklearn.preprocessing import StandardScaler
# Create an instance of StandardScaler class
scaler = StandardScaler()

# Fit and transform the data using the scaler instance
df_norm = scaler.fit_transform(df)
df_norm = pd.DataFrame(data= df_norm, columns = df_col)


#Separate the datasets
X=df.drop(columns=['Price'],axis=1)
y=df['Price']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=4)


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
# def predict(ml_model):
#     print('Model is: {}'.format(ml_model))
#     model= ml_model.fit(X_train,y_train)
#     print("Training score: {}".format(model.score(X_train,y_train)))
#     predictions = model.predict(X_test)
#     print("Predictions are: {}".format(predictions))
#     print('\n')
#     r2score=r2_score(y_test,predictions) 
#     print("r2 score is: {}".format(r2score))
          
#     print('MAE:{}'.format(mean_absolute_error(y_test,predictions)))
#     print('MSE:{}'.format(mean_squared_error(y_test,predictions)))
#     print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,predictions))))

regressor = RandomForestRegressor()
regressor.fit (X_train,y_train)


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))