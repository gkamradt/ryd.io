from flask import Flask
from flask import request
from flask import render_template
from flask import jsonify
from bigquery_request import BigQuery
from predict_location import LocationPredict
import json
from time import strftime

app = Flask(__name__)
app.business_location = None
app.previous_location_request = None
app.bq = BigQuery(project_id='###', \
            service_account='###', \
            key_location='###')
app.cluster_img_dict = {0:'static/dist/img/cluster_0.jpg',\
                        1:'static/dist/img/cluster_1.jpg',\
                        2:'static/dist/img/cluster_2.jpg',\
                        3:'static/dist/img/cluster_3.jpg',\
                        4:'static/dist/img/cluster_4.jpg',\
                        5:'static/dist/img/cluster_5.jpg'}
#Location Predict object place holder
app.lp = None

# OUR HOME PAGE
#============================================
@app.route('/')
def welcome():
    myname = "Greg"
    return render_template('index.html', data=myname)


@app.route('/userinput', methods=['POST'])
def userinput():
    # Recieve data from the user input form. We don't want to query bigquery again if we are looking at the same location
    # We will only query bigquery again if the location changes.

    r = json.loads(request.data)

    lat, long_ = r['lat_long'][0], r['lat_long'][1]
    hour_select = str(r['hour_select'])
    days_from_today_select = str(r['date_select'])
    app.business_location = r['lat_long']
    print hour_select, type(hour_select)

    if app.business_location == app.previous_location_request:
        # When the location is the same as before. Don't Query again
        rides_predicted = app.lp.predict_rides_by_hour_via_cluster(days_out_to_predict=days_from_today_select, hour=hour_select)
        cluster_num = app.lp.nearest_cluster((lat, long_))
    else:
        # When you need to load new data
        app.previous_location_request = app.business_location

        df =  app.bq.query_location(lat, long_)
        app.lp = LocationPredict(df)
        rides_predicted = app.lp.predict_rides_by_hour_via_cluster(days_out_to_predict=days_from_today_select, hour=hour_select)
        cluster_num = app.lp.nearest_cluster((lat, long_))


    print 'rides_predicted', rides_predicted, \
            'rides_predicted_count', len(rides_predicted['coordinates']),\
            'nearest_cluster', app.lp.nearest_cluster((lat, long_)),\
            'predict_date', app.lp.date_to_predict.strftime('%m/%d/%Y')
    return jsonify({'rides_predicted':rides_predicted, \
                    'rides_predicted_count': len(rides_predicted['coordinates']),\
                    'nearest_cluster': cluster_num,\
                    'predict_date': app.lp.date_to_predict.strftime('%m/%d/%Y'),\
                    'cluster_img_path': app.cluster_img_dict[cluster_num]})

@app.route('/map_predict')
def map_predict():
    return render_template('map_predict.html')

@app.route('/technical_summary')
def technical_summary():
    return render_template('technical_summary.html')

@app.route('/cluster_map')
def cluster_map():
    return render_template('cluster_map.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)
