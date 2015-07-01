from bigquery import get_client
import pandas as pd
import time


class BigQuery(object):
    """
    Get ride information via bigquery for a specific lat and long location.
    This will get run each time a user changes a location.

    """
    def __init__(self, project_id, service_account, key_location):
        self.x = 'Hello'
        # BigQuery project id as listed in the Google Developers Console.
        self.project_id = project_id.lower()
        # Service account email address as listed in the Google Developers Console.
        self.service_account = service_account
        # PKCS12 or PEM key provided by Google.
        self.key = key_location

        self.client = get_client(self.project_id, service_account=self.service_account,
                            private_key_file=self.key, readonly=True)

    def query_location(self, center_lat, center_long, radius_multiple=1):
        """
        INPUT:
        -center_lat(float) - User defined - Latitude of coordinate in question
        -center_long(float) - User defined - Longitude of coordinate in question
        -radius_multiple(int) - User defined - Distance of a radius you would like. 1 Radius Multiple ~ 1 half a NYC Block
        
        OUTPUT:
        -Pandas DataFrame - User defined - Will return all rides in a given location for the entire year.

        DOC:
        -This method will query BigQuery and return all rides that meet our location requirements. Each query has BigQuery run through ~30GB of data
        """

        # Set your radius
        center_lat, center_long = center_lat, center_long
        radius = float(radius_multiple) * .000757

        # Set your lat/longs
        lat_top = center_lat + abs(radius/1.3)
        lat_bottom = center_lat - abs(radius/1.3)
        long_left = center_long - abs(radius)
        long_right = center_long + abs(radius)

        query = """
        SELECT
          pickup_datetime, dropoff_datetime, passenger_count, trip_distance, pickup_longitude, pickup_latitude, dropoff_longitude,  dropoff_latitude
        FROM
          nycdata_trips.trip_data_g
        WHERE
              FLOAT(dropoff_latitude) < %s
          AND FLOAT(dropoff_latitude) > %s
          AND FLOAT(dropoff_longitude) > %s
          AND FLOAT(dropoff_longitude) < %s
        """ % (lat_top, lat_bottom, long_left, long_right)

        # Submit an async query.
        job_id, _results = self.client.query(query)

        # Sleeping to allow the query to have time to run. I usually get an unfishedquery exception if I don't
        time.sleep(5)

        # Retrieve the results.
        try:
            results = self.client.get_query_rows(job_id)
            return pd.DataFrame(results)
        except Exception as e:
            # Wait longer if you get an issue

            time.sleep(15)
            results = self.client.get_query_rows(job_id)
        return pd.DataFrame(results)

    def query_open(self, query_string):
        """
        INPUT:
        query_string(string) - User Defined - Take raw custom SQL query and query the database in bigquery

        OUTPUT:
        df(Pandas DataFrame) - Results from query
        """
        query = query_string
        # Submit an async query.
        job_id, _results = client.query(query)

        # Check if the query has finished running.
        complete, row_count = client.check_job(job_id)

        # Retrieve the results.
        results = client.get_query_rows(job_id)
        df = pd.DataFrame(results)
        return df

if __name__ == '__main__':
    bq = BigQuery(project_id='###', \
                    service_account='###', \
                    key_location='###')
    response = bq.query_location(40.738019, -73.996361)
    print len(response)