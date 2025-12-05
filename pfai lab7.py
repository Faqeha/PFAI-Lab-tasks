from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

@app.route("/")
def home():
    return jsonify({"message": "Weather API is running. Use /weather?city=CityName"})

@app.route("/weather", methods=["GET"])
def get_weather():
    city = request.args.get("city")

    if not city:
        return jsonify({"error": "City name required. Example: /weather?city=London"}), 400

    url = f"{BASE_URL}?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code != 200:
        return jsonify({"error": "City not found or API error"}), 404

    data = response.json()

    weather_data = {
        "city": data["name"],
        "temperature_celsius": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "weather_condition": data["weather"][0]["description"],
        "wind_speed_kmh": data["wind"]["speed"]
    }

    return jsonify(weather_data)

if __name__ == "__main__":
    app.run(debug=True)
