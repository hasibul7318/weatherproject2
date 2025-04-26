from django.shortcuts import render


import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from datetime import datetime, timedelta
import pytz
import os

API_KEY ='dcba9688a778ca233f5be09a0ea0dc72'
BASE_URL ='https://api.openweathermap.org/data/2.5/'


def get_current_weather(city):
    """
    Fetches current weather data for a given city using an API.

    :param city: str, name of the city
    :return: dict, weather details or error message
    """
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code != 200:
        return {'error': f"Error {response.status_code}: {response.json().get('message', 'Invalid request')}"}

    data = response.json()

    return {
        'city': data.get('name', 'N/A'),
        'current_temp':round(data['main'].get('temp',0)),
        'temperature': round(data['main'].get('temp', 0)),
        'feels_like': round(data['main'].get('feels_like', 0)),
        'temp_min': round(data['main'].get('temp_min', 0)),
        'temp_max': round(data['main'].get('temp_max', 0)),
        'humidity': data['main'].get('humidity', 0),
        'wind_speed': data['wind'].get('speed', 0),
        'wind_gust_dir': data['wind'].get('deg', 0),
        'pressure': data['main'].get('pressure', 0),
        'wind_gust_speed': data['wind'].get('gust', 0),
        'description': data['weather'][0].get('description', 'N/A') if data.get('weather') else 'N/A',
        'icon': data['weather'][0].get('icon', '') if data.get('weather') else '',
        'country': data['sys'].get('country', 'N/A'),
        'clouds': data['clouds'].get('all', 0),  # Fixed potential KeyError
        'visibility': data.get('visibility', 0)  # Fixed potential KeyError
    }




def read_historical_data(filename):
    """
    Reads a CSV file, removes missing values and duplicates, and returns a cleaned DataFrame.

    :param filename: str, path to the CSV file
    :return: pandas DataFrame
    """
    df = pd.read_csv(filename)  # Load CSV file into DataFrame
    df = df.dropna()  # Remove rows with missing values
    df = df.drop_duplicates()  # Remove duplicate rows

    return df




def prepare_data(data):
    """
    Prepares the dataset for training.
    - Encodes categorical variables.
    - Selects relevant features and target variables.
    """

    # Create a LabelEncoder instance
    le = LabelEncoder()

    # Encode categorical features
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

    # Define the feature variables (X) and target variable (y)
    X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure']]
    y = data['RainTomorrow']

    return X, y, le





def train_rain_model(x, y):
    """
    Trains a RandomForestClassifier to predict rain and evaluates its performance.

    :param x: pandas DataFrame or numpy array (features)
    :param y: pandas Series or numpy array (target variable)
    :return: Trained model
    """
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    print("Mean Squared Error for Rain Model:", mean_squared_error(y_test, y_pred))

    return model





def prepare_regression_data(data, feature):
    """
    Prepares the dataset for time-series regression, shifting the feature column.

    :param data: pandas DataFrame
    :param feature: str, column name to use for regression
    :return: Tuple (X, y) as numpy arrays
    """
    x, y = [], []  # Initialize lists for features and target values

    for i in range(len(data) - 1):
        x.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])

    # Convert lists to numpy arrays
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    return x, y





def train_regression_model(x, y):
    """
    Trains a RandomForestRegressor model on the given dataset.

    :param x: numpy array or pandas DataFrame (features)
    :param y: numpy array or pandas Series (target variable)
    :return: Trained RandomForestRegressor model
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x, y)

    return model





def predict_future(model, current_value, steps=5):
    """
    Predicts future values using a trained regression model.

    :param model: Trained regression model (e.g., RandomForestRegressor)
    :param current_value: The latest known value to start predictions
    :param steps: Number of future predictions to make (default: 5)
    :return: List of predicted future values
    """
    predictions = [current_value]

    for _ in range(steps):
        next_value = model.predict(np.array([[predictions[-1]]]))  # Ensure correct input shape
        predictions.append(next_value[0])  # Extract scalar value

    return predictions[1:]  # Return only future predictions





        
#import pandas as pd
#import numpy as np
#import pytz
#from datetime import datetime, timedelta


def weather_view(request):
    city = "Kolkata"  # Default city

    if request.method == "POST":
        city = request.POST.get("city", "New Delhi")

    # Fetch current weather
    current_weather = get_current_weather(city)

    if 'error' in current_weather:
        print(current_weather['error'])
        return render(request, "weather.html", {"error": "Weather data unavailable"})

    # Load historical weather data
    csv_path = os.path.join(settings.BASE_DIR, 'forecast', 'static', 'weather.csv')
     df = pd.read_csv(csv_path)
    historical_data = read_historical_data(csv_path)

    # Prepare data for training
    x, y, le = prepare_data(historical_data)
    rain_model = train_rain_model(x, y)

    # Get wind direction
    wind_deg = current_weather.get("wind_gust_dir", 0) % 360

    # Determine wind compass direction
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75), ("N", 348.75, 360)
    ]

    compass_direction = next(
        point for point, start, end in compass_points if start <= wind_deg < end
    )

    # Encode wind direction
    compass_direction_encoded = (
        le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1
    )

    # Prepare current weather data for prediction
    current_data = {
        "MinTemp": current_weather.get("temp_min", 0),
        "MaxTemp": current_weather.get("temp_max", 0),
        "WindGustDir": compass_direction_encoded,
        "WindGustSpeed": current_weather.get("wind_gust_speed", 0),
        "Humidity": current_weather.get("humidity", 0),
        "Pressure": current_weather.get("pressure", 1013),
    }

    current_df = pd.DataFrame([current_data])
    current_df.fillna(0, inplace=True)

    # Predict rain
    rain_prediction = rain_model.predict(current_df)[0]

    # Train regression models
    X_temp, y_temp = prepare_regression_data(historical_data, "Temp")
    X_hum, y_hum = prepare_regression_data(historical_data, "Humidity")
    temp_model = train_regression_model(X_temp, y_temp)
    hum_model = train_regression_model(X_hum, y_hum)

    # Predict future weather
    future_temp = predict_future(temp_model, current_weather.get("temp_min", 0))
    future_humidity = predict_future(hum_model, current_weather.get("humidity", 0))

    # Time formatting
    timezone = pytz.timezone("Asia/Kolkata")
    now = datetime.now(timezone)
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

    # Assign predictions
    time1, time2, time3, time4, time5 = future_times
    temp1, temp2, temp3, temp4, temp5 = future_temp
    hum1, hum2, hum3, hum4, hum5 = future_humidity

    context = {
        "location": city,
        "current_temp": current_weather.get("current_temp", 0),
        "MinTemp": current_weather.get("temp_min", 0),
        "MaxTemp": current_weather.get("temp_max", 0),
        "feels_like": current_weather.get("feels_like", 0),
        "humidity": current_weather.get("humidity", 0),
        "clouds": current_weather.get("clouds", 0),
        "description": current_weather.get("description", "N/A"),
        "city": current_weather.get("city", city),
        "country": current_weather.get("country", "Unknown"),
        "time": datetime.now(),
        "date": datetime.now().strftime("%B %d, %Y"),
        "wind": current_weather.get("wind_gust_speed", 0),
        "pressure": current_weather.get("pressure", 1013),
        "visibility": current_weather.get("visibility", 10000),
        "time1": time1, "time2": time2, "time3": time3, "time4": time4, "time5": time5,
        "temp1": f"{round(temp1,1)}", "temp2": f"{round(temp2,1)}",
        "temp3": f"{round(temp3,1)}", "temp4": f"{round(temp4,1)}", "temp5": f"{round(temp5,1)}",
        "hum1": f"{round(hum1,1)}", "hum2": f"{round(hum2,1)}",
        "hum3": f"{round(hum3,1)}", "hum4": f"{round(hum4,1)}", "hum5": f"{round(hum5,1)}",
    }

    return render(request, "weather.html", context)
