import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd
import yfinance as yf
import math
import datetime as dt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import requests
from sklearn.metrics import mean_squared_error
from textblob import TextBlob
import json

config = json.loads(open("config.json").read())



class Predictor:
    
    def __init__(self,company_symbol,days_to_calculate):
        self.company_symbol = company_symbol
        self.start_date_train = dt.datetime(2013, 1, 1)
        self.end_date_train = self.end_date_test = dt.datetime.now()
        self.start_date_test = dt.datetime(2022, 5, 1)
        self.days_to_calculate = days_to_calculate

        self.current_price, self.long_name = self.get_stock()
        self.company_data_train = self.get_train_data()
        self.company_data_test = self.get_test_data()
        self.scaler = self.create_scaler()
        self.scaled_data = self.scale_data()
        self.x_train, self.y_train = self.prepare_training_data()
        self.x_test = self.prepare_test_data()
        self.model = self.create_model()

        self.train_model()

        self.predicted_prices = self.predict()
        self.actual_prices = self.company_data_test['Close'].values
        
        self.headlines,self.average_sentiment = self.get_headlines_and_sentiment()
        self.average_daily_change = self.get_average_daily_change()

        self.create_graph()
        self.save_dataframe()
        self.calculate_accuracy()

    def calculate_accuracy(self):
        absolute_percentage_errors = np.abs((self.actual_prices - self.predicted_prices) / self.actual_prices)
        mean_absolute_percentage_error = np.mean(absolute_percentage_errors)
        accuracy = 100 - (mean_absolute_percentage_error * 100)
        return accuracy
    
    def create_model(self):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model


    def get_train_data(self):
        return yf.download(self.company_symbol, start=self.start_date_train, end=self.end_date_train)
    
    def get_test_data(self):
        return yf.download(self.company_symbol, start=self.start_date_test, end=self.end_date_test)
        
    def create_scaler(self):
        return MinMaxScaler(feature_range=(0, 1))

    def scale_data(self):
        return self.scaler.fit_transform(self.company_data_train['Close'].values.reshape(-1, 1))

    def prepare_training_data(self):
        x_train = []
        y_train = []

        for x in range(self.days_to_calculate, len(self.scaled_data)):
            x_train.append(self.scaled_data[x - self.days_to_calculate:x, 0])
            y_train.append(self.scaled_data[x, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        return x_train, y_train

    def prepare_test_data(self):
        total_dataset = pd.concat((self.company_data_train['Close'], self.company_data_test['Close']), axis=0)

        model_inputs = total_dataset[len(total_dataset) - len(self.company_data_test) - self.days_to_calculate:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = self.scaler.transform(model_inputs)

        x_test = []
        for x in range(self.days_to_calculate, len(model_inputs)):
            x_test.append(model_inputs[x - self.days_to_calculate:x, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test

    def train_model(self):
        self.model.fit(self.x_train, self.y_train, epochs=25, batch_size=32)

    def predict(self):
        predicted_prices = self.model.predict(self.x_test)
        predicted_prices = self.scaler.inverse_transform(predicted_prices)
        return predicted_prices
    
    def create_graph(self):
        plt.figure(figsize=(15,8))
        plt.plot(self.company_data_test.index, self.actual_prices, color="blue", label=f"Actual {self.company_symbol} Price")
        plt.title(f"{self.company_symbol} Share Price")
        plt.xlabel('Date')
        plt.ylabel(f'{self.company_symbol} Share Price')
        plt.legend()
        plot_filename = f'static/{self.company_symbol}_plot.png'
        plt.savefig(plot_filename)

    def save_dataframe(self):
        predicted_df = pd.DataFrame(self.predicted_prices, columns=['Predicted Price'])
        actual_df = pd.DataFrame(self.actual_prices, columns=['Actual Price'])
        merged_df = pd.concat([actual_df, predicted_df], axis=1)
        merged_df.index = self.company_data_test.index
        csv_filename = f'static/{self.company_symbol}_predictions.csv'
        merged_df.to_csv(csv_filename)

    def get_stock(self):
        stock = yf.Ticker(self.company_symbol)
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        long_name = stock.info['longName']
        return current_price, long_name
    
    def get_headlines_and_sentiment(self):
        api_key = config['api_key']
        date = self.end_date_test - dt.timedelta(days=30)
        url = f'https://newsapi.org/v2/everything?q={self.long_name}&from={date}&sortBy=popularity&apiKey={api_key}'

        try:
            response = requests.get(url)
            newsdata = response.json()
            headlines = [article["title"] for article in newsdata["articles"]]
            dates = [article["publishedAt"][:10] for article in newsdata["articles"]]
            df3 = pd.DataFrame({"headline": headlines, "Date": dates})
            df3["sentiment"] = ""
            for i in range(len(headlines)):
                text_blob = TextBlob(headlines[i])
                sentiment_polarity = text_blob.sentiment.polarity
                if sentiment_polarity > 0:
                    df3["sentiment"][i] = sentiment_polarity
                elif sentiment_polarity < 0:
                    df3["sentiment"][i] = sentiment_polarity
                else:
                    df3["sentiment"][i] = sentiment_polarity

            average_sentiment = df3["sentiment"].mean()
            return headlines,average_sentiment
        
        except Exception as e:
            print(f"Error in fetching Headlines from API: {e}")

        
    def get_average_daily_change(self):
        daily_change = self.company_data_train['Adj Close'].diff()
        average_daily_change = daily_change.mean()
        return average_daily_change
    
    def get_all_data_for_web(self):
        predicted_price = self.predicted_prices[-1][0] + (math.lgamma(self.average_daily_change + self.average_sentiment))
        return self.long_name, predicted_price, self.current_price, f'static/{self.company_symbol}_plot.png', self.headlines