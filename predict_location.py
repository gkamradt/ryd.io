import statsmodels.api as sm
import pandas as pd
from scipy.stats import rv_discrete
import scipy.stats as stats
import numpy as np
import datetime
import geojson

class LocationPredict(object):
    """
    INPUT:
    -dataset(Pandas DataFrame) - [From BigQuery.py]
    -grouped_by_day(boolean) - True if your data is already grouped by day
    -nonseasonal_order(tuple) - Nonseasonal parameters into the SARIMAX model 
    -seasonal_order(tupe) - seasonal parameters into the SARIMAX model

    OUTPUT:
    Predicted rides in your choice of format: Geojson, csv, Pandas DataFrame

    DOC:
    -LocationPredict takes in a Pandas DataFrame and returns predictions of future rides
    -Can predict a simple daily total, or can output a specific hour
    """

    def __init__(self, dataset, grouped_by_day=False, nonseasonal_order=(1,1,1), seasonal_order=(0,1,1,7),\
                cluster_dict = "data/manhatten_day_hour_cluster_normed.csv", cluster_labeling="data/manhatten_lat_long_clusters"):
        # Seasonal parameters
        self.nonseasonal_order = nonseasonal_order
        self.seasonal_order = seasonal_order
        
        # Object data set. Two versions, one to work with, one to reference with later
        self.data = dataset
        self.df_raw = dataset

        # Create a range of dates to index later
        self.date_range_list = pd.date_range(start='2013-01-01', end='2015-12-31')

        # Variable for holder of date user will put in later
        self.date_to_predict = None

        # Kernerl blah. To be filled in by function X
        self.kernel = None

        # Cluster Label holder that model will predict
        self.cluster_label = None

        # Empty list that will hold different distributions for days of the week
        self.dayofweek_hour_dist = []

        # If the user gives you total rides and you need to group them by day
        if not grouped_by_day:
            self.group_by_day()

        # Initalize your hourly distributions and KDE distribution
        self.init_hourly_dist()
        self.init_multi_gaussian_kde()

        # Train your SARIMAX model based off the grouped by day self.df
        self.train_model()

        # The maximum/minimum for both latitude and longitude
        self.lat_top = float(self.df_raw.dropoff_latitude.max())
        self.lat_bottom = float(self.df_raw.dropoff_latitude.min())
        self.long_left = float(self.df_raw.dropoff_longitude.min())
        self.long_right = float(self.df_raw.dropoff_longitude.max())

        #cluster dict is the clustering of different locations. Will apply hourly dist after
        self.cluster_dict = pd.read_csv(cluster_dict, index_col=0)
        self.cluster_labeling = pd.read_csv(cluster_labeling, index_col = 0)
        
    def init_hourly_dist(self):
        """
        INPUT:
        -None

        OUTPUT:
        -None

        DOC:
        -init_hourly_dist takes

        """
        self.dayofweek_hour_dist = []
        for day in range(7):
            df_day = self.df_raw[self.df_raw.dropoff_datetime.dt.dayofweek == day]['dropoff_datetime']
            hourly_dist = df_day.dt.hour.value_counts().sort_index() / len(df_day)
            self.dayofweek_hour_dist.append(rv_discrete(values=(hourly_dist.index, hourly_dist.values)))

    def init_multi_gaussian_kde(self):
        """
        INPUT:
        -None

        OUTPUT:
        -Self.kernel - Multivariate KDE that we can resample from later

        DOC:
        -Simply initalizing a KDE based off of previous lat and longs of the dataframe

        """
        np.random.seed(seed=1)
        lats = self.df_raw.dropoff_latitude.values.astype(float)
        longs = self.df_raw.dropoff_longitude.values.astype(float)
        xmin = longs.min()
        xmax = longs.max()
        ymin = lats.min()
        ymax = lats.max()

        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([longs, lats])
        self.kernel = stats.gaussian_kde(values)
        return self.kernel

    def train_model(self):
        """
        INPUT:
        -None

        OUTPUT:
        -A fitted SARIMAX model

        DOC:
        -The SARIMAX model takes in a series of points and outputs a fitted model
        -These points happen to be the number of rides that were grouped by date

        """
        self.data = self.data['visits']
        self.model = sm.tsa.SARIMAX(self.data, order=self.nonseasonal_order, seasonal_order=self.seasonal_order).fit()

    def predict_date_total(self, date):
        """
        INPUT:
        -date(string)

        OUTPUT:
        day_prediction(int) - number of rides predicted for a particular day

        DOC:
        -Based off the model, call model.predict to make a prediction for a specified date in the future
        -The model can't take in a string so you have to convert input date to a int (num of days since Jan 1, 2013)

        """
        predict_index = self.date_range_list.searchsorted(date)
        day_prediction = self.model.predict(start=predict_index, end=predict_index)[0]
        return int(day_prediction)

    def predict_date_hour(self, date, hour):
        """
        INPUT:
        -date(string)
        -hour(int)

        OUTPUT:
        -Total number of rides predicted for a specified hour
        """

        # Get total for the day
        total_for_the_day = self.predict_date_total(date)

        # Find out what day of the week you are forcasting
        day_of_week = pd.to_datetime(date).dayofweek

        # Sample your rvs_discrete "day prediction total" times. You're actually getting a daily hourly distribution total at this point
        sample_hour_distribution = list(self.dayofweek_hour_dist[day_of_week].rvs(size=total_for_the_day))

        # Return only the number of items that meet your critera of hour specified
        rides_in_hour = sample_hour_distribution.count(hour)

        return rides_in_hour

    def predict_rides_by_hour(self, date):
        """
        INPUT:
        -date(string)

        OUTPUT:
        -df_rides(DataFrame) - DataFrame of predicted rides
        """

        # Get your temporary lists ready
        rides_time_df = []
        lat_list = []
        long_list = []

        # Get your total amount predicted for the date specified
        total_for_the_day = self.predict_date_total(date)

        # Get day of the week you care about
        day_of_week = pd.to_datetime(date).dayofweek

        # Sample your rvs_discrete "day prediction total" times. You're actually getting a daily hourly distribution total at this point
        # Example ourput [5,5,4,5,7,5,23,4,5,7,8,7,5,4,3] Representing an hour that a ride came through
        rides_by_hour = self.dayofweek_hour_dist[day_of_week].rvs(size=total_for_the_day)

        # Create a loop that goes through each entry in rides_by_hour and attaches a lat/long and random minute/second combo to each hour
        for ride_hour in rides_by_hour:

            # Add np.random.uniform minute to each hour
            rides_time_df.append(pd.to_datetime('%s %s:%s' % (date, ride_hour, np.random.uniform(low=0, high=60, size = 1)[0])))

            # Lat/Long prediction. Then split up the tuple
            lat_long = self.predict_lat_long()
            lat_list.append(lat_long[1][0])
            long_list.append(lat_long[0][0])

        # Create a dictionary out of all of the lists you just made. This is the middle step to outputing a DataFrame
        d = {'time': rides_time_df, 'latitude': lat_list, 'longitude': long_list, 'type': 'predict'}

        # Create a DataFrame from your dictionary
        df_rides = pd.DataFrame(d, columns=['time', 'latitude', 'longitude', 'type'])
 
        return df_rides

    def predict_rides_by_hour_via_cluster(self, days_out_to_predict=None, hour=None, date_to_predict=None, date_today=datetime.date.today()):
        """
        Filter self.data just to return all the points up until yesterday.
        This function is messy. Can be made more pretty later.
        """
        """
        INPUT:
        -days_out_to_predict(int) - User specified - Number of days in the future to predict (tomorrow = +1, a week from today=+7)
        -hour(int) - User specified - Hour of day to predict
        -date_to_predict(string) - Optional - User specified
        -date_today(datetime) - Easy way to update the date each time. This will ensure that we only train the model on previous data.

        OUTPUT:
        -Geojson to plot on leaflet for webapp

        DOC:
        -This method is very similar to self.predict_rides_by_hour, however, the hourly distribution is calculated via the cluster hourly distribution.
        """

        # Temporary lists
        rides_time_df = []
        lat_list = []
        long_list = []

        # If the user actually put in a date. Most of the time this will be None
        if date_to_predict == None:

            # were always going to guess 2013
            predict_year = 2013

            #Month of prediction
            predict_month = (datetime.date.today() + datetime.timedelta(days=int(days_out_to_predict))).month

            #Day prediction
            predict_day = (datetime.date.today() + datetime.timedelta(days=int(days_out_to_predict))).day

            # The same day but in 2013
            today_in_2013 = datetime.date(2013, datetime.date.today().month, datetime.date.today().day)
            self.date_to_predict = today_in_2013 + datetime.timedelta(days=int(days_out_to_predict))



        #cutting off the df so that it is trained on only previous data
        date_to_filter_to = datetime.date(2013, day=date_today.day, month=date_today.month) - datetime.timedelta(days=1)
        filtered_df =  self.data[self.data.index <= str(date_to_filter_to)]

        # Train your model
        model = sm.tsa.SARIMAX(filtered_df.values, order=self.nonseasonal_order, seasonal_order=self.seasonal_order, enforce_invertibility=False).fit()

        # Getting date number to feed into model.predict
        predict_day_num = self.date_range_list.searchsorted(self.date_to_predict)

        # to ensure that we never ask for a prediction start date that is past the trained data
        if predict_day_num > len(filtered_df):
            start_num = len(filtered_df)
        else:
            start_num = predict_day_num


        # set the lat/long of request
        point_lat = self.df_raw.dropoff_latitude.values.astype(float).mean()
        point_long = self.df_raw.dropoff_longitude.values.astype(float).mean()

        # Models prediction of a certain day
        day_total_prediction = int(model.predict(start=start_num, end=predict_day_num)[0][-1])

        # See what cluster the point in question is closest to
        self.cluster_label = self.nearest_cluster((point_lat, point_long))

        # Getting the proportion of rides we predict via the nearest cluster  
        hour_dropoff_proportion = self.cluster_dict.iloc[self.cluster_label]['h%s' % hour]

        # Getting the actual number of rides
        rides_arriving_in_hour =  int(day_total_prediction * hour_dropoff_proportion)

        for ride_hour in range(rides_arriving_in_hour):

            # Add np.random.uniform minute to each hour
            rides_time_df.append(pd.to_datetime('%s %s:%s' % (date_to_predict, hour, np.random.uniform(low=0, high=60, size = 1)[0])))

            # Lat/Long prediction. Then split up the tuple
            lat_long = self.predict_lat_long()
            lat_list.append(lat_long[1][0])
            long_list.append(lat_long[0][0])

        # Create a dictionary out of all of the lists you just made. This is the middle step to outputing a DataFrame
        d = {'time': rides_time_df, 'latitude': lat_list, 'longitude': long_list, 'type': 'predict'}

        # Create a DataFrame from your dictionary
        df_rides = pd.DataFrame(d, columns=['time', 'latitude', 'longitude', 'type'])

        # return df_rides.to_dict(orient='records')
        return geojson.MultiPoint(zip(*(df_rides.latitude, df_rides.longitude)))


    def predict_lat_long(self):
        """
        INPUT:
        -None

        OUTPUT:
        -Lat/Long prediction

        DOC:
        -Just resampling the KDE that is already made

        """
        lat_long = self.kernel.resample(1)
        return lat_long

    def group_by_day(self):
        """
        INPUT:
        -None

        OUTPUT:
        -None

        DOC:
        -Takes a raw df of rides and groups them by date. This output will get fed into the SARIMAX model

        """

        # Create date range
        self.data.dropoff_datetime = pd.to_datetime(self.data.dropoff_datetime)
        dates = pd.date_range(start=self.data.dropoff_datetime.dt.date.min(), end=self.data.dropoff_datetime.dt.date.max(), freq="D")

        # Group by date and count the entries. Then sort the index so we can search it easily
        self.data = self.data.dropoff_datetime.dt.date.value_counts().sort_index()

        # Make and clean up your df
        self.data = pd.DataFrame(data=self.data, index=dates)
        self.data.columns = ['visits']

        return

    def nearest_cluster(self, (lat_long)):
        """
        INPUT:
        -lat_long(tuple) - User specified lat and long

        OUTPUT:
        -cluster_label

        DOC:
        -This method will search through your cluster list and return the label of the cluster that you are closest to.

        """

        # Create temp variable
        my_lat_long = lat_long

        # Empty list that you'll fill with distances later
        distances_from_point = []

        # Lat and long clusters
        lat_long_clusters = self.cluster_labeling

        # Fill your empty list with distances
        for cluster_point in zip(*(lat_long_clusters.lat, lat_long_clusters.long)):
            distances_from_point.append(np.linalg.norm(np.array(cluster_point) - np.array(my_lat_long)))

        # Return the label of the cluster you are closest to
        cluster_label = lat_long_clusters.iloc[np.argmin(distances_from_point)]['label']
        return cluster_label

    def export_rides(self, date, type_of_export, path="", csv=False):
        """
        INPUT:
        -date(string) - User defined
        -type_of_export(string) - User defined
        -path(string) - User defined

        OUTPUT:
        -CSV or JSON file

        DOC:
        -The method goes through logic to decide what kind of file the user wants and exports accordingly.
        -This method is useful when visualizing predictions vs actuals and needing to write either/both of them to a file.

        """
        """
        Takes: date, type_of_export, path
        if no path is given then a df will be returned, not written to file

        """
        if type_of_export == 'predictions':
            predicted_list_of_rides_by_hour_df = self.predict_rides_by_hour(date)
            predicted_list_of_rides_by_hour_df.reset_index(inplace=True, drop=True)
            if path == "":
                return predicted_list_of_rides_by_hour_df
            else:
                if csv:
                    predicted_list_of_rides_by_hour_df.to_csv(path, orient='index')
                else:
                    predicted_list_of_rides_by_hour_df.to_json(path, orient='index')
                return
        elif type_of_export == 'actuals':
            actual_list_of_rides_by_hour_df = self.df_raw[self.df_raw.dropoff_datetime.dt.date == pd.to_datetime(date).date()]
            actual_list_of_rides_by_hour_df = actual_list_of_rides_by_hour_df[['dropoff_datetime', 'dropoff_latitude', 'dropoff_longitude']]
            actual_list_of_rides_by_hour_df['type'] = 'actual'
            actual_list_of_rides_by_hour_df.columns = ['time', 'latitude', 'longitude', 'type']
            actual_list_of_rides_by_hour_df.reset_index(inplace=True, drop=True)
            if path == "":
                return actual_list_of_rides_by_hour_df
            else:
                if csv:
                    predicted_list_of_rides_by_hour_df.to_csv(path)
                else:
                    predicted_list_of_rides_by_hour_df.to_json(path, orient='index')
                return
        elif type_of_export == 'both':
            pred_act_df = pd.concat([self.export_rides(date, 'predictions'), self.export_rides(date, 'actuals')])
            pred_act_df.reset_index(inplace=True, drop=True)
            if path == "":
                return pred_act_df
            else:
                if csv:
                    pred_act_df.to_csv(path, orient='index')
                else:
                    pred_act_df.to_json(path, orient='index')
                return
        else:
            print "invalid type, choose 'predictions', 'actuals', or 'both'" 

if __name__ == '__main__':
    df = pd.read_csv('data/corner_data/broadway_spring_main.csv')
    lp = LocationPredict(dataset=df, grouped_by_day=False)
    print lp.predict_date_total('2013-01-02')