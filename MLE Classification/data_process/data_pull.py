# Importing required libraries
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging 
import os

# Logging 
logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

url = 'https://en.wikipedia.org/wiki/Iris_flower_data_set'

logging.info('Extracting data from URL')
html = requests.get(url).content
df_list = pd.read_html(html)

data = pd.DataFrame(df_list[0])
data.set_index(data['Dataset order'], inplace = True)
data = data.drop('Dataset order', axis = 1)

logging.info('Encoding target variable')
enc = LabelEncoder()
data['Species'] = enc.fit_transform(data['Species'])

logging.info('Splitting into train and test')
train, test, ytrain, ytest = train_test_split(data.iloc[:,:-1], data.Species, test_size = 0.2, shuffle = True)

data_path = os.path.join(os.getcwd(), 'data_process')
logging.info(f'Saving files in {data_path}')

train = train.to_csv(os.path.join(data_path,'train.csv'), index = False)
ytrain = ytrain.to_csv(os.path.join(data_path,'ytrain.csv'), index = False)
test = test.to_csv(os.path.join(data_path,'test.csv'), index = False)
ytest = ytest.to_csv(os.path.join(data_path,'ytest.csv'), index = False)

logging.info('Files saved')