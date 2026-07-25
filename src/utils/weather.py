import requests
from typing import Optional
from dataclasses import dataclass


@dataclass
class WeatherData:
    temperature_c: float
    humidity_pct: float
    wind_speed_kmh: float
    description: str
    rain_mm: float
    location: str


class WeatherAdvisor:
    """Retrieves localized weather data and generates treatment protocols.

    Combines real-time environmental data with disease diagnosis to
    produce weather-aware treatment recommendations for cotton crops.
    """

    TREATMENT_PROTOCOLS = {
        "Bacterial_Blight": {
            "high_humidity": "Apply copper-based bactericide. Improve drainage.",
            "dry": "Remove infected plants. Apply streptomycin spray.",
            "default": "Monitor spread. Apply Bordeaux mixture preventively.",
        },
        "CLCuD": {
            "high_humidity": "Apply systemic insecticide for whitefly control.",
            "dry": "Remove infected plants. Install yellow sticky traps.",
            "default": "Use resistant varieties. Apply neem oil spray.",
        },
        "Alternaria_Leaf_Spot": {
            "high_humidity": "Apply azoxystrobin fungicide. Increase plant spacing.",
            "dry": "Remove affected leaves. Apply chlorothalonil.",
            "default": "Rotate crops. Apply mancozeb fungicide.",
        },
        "Fusarium_Wilt": {
            "high_humidity": "Apply Trichoderma biocontrol agent. Reduce irrigation.",
            "dry": "Remove infected plants. Apply carboxin fungicide to soil.",
            "default": "Use resistant rootstock. Solarize soil.",
        },
        "healthy": {
            "default": "Continue regular monitoring. No treatment needed.",
        },
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def fetch_weather(self, lat: float, lon: float) -> Optional[WeatherData]:
        if not self.api_key:
            return WeatherData(
                temperature_c=28.0,
                humidity_pct=65.0,
                wind_speed_kmh=12.0,
                description="clear sky",
                rain_mm=0.0,
                location="simulated",
            )
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",
            }
            resp = requests.get(self.base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return WeatherData(
                temperature_c=data["main"]["temp"],
                humidity_pct=data["main"]["humidity"],
                wind_speed_kmh=data["wind"]["speed"] * 3.6,
                description=data["weather"][0]["description"],
                rain_mm=data.get("rain", {}).get("1h", 0.0),
                location=data.get("name", "unknown"),
            )
        except Exception:
            return None

    def classify_conditions(self, weather: WeatherData) -> str:
        if weather.humidity_pct > 80 or weather.rain_mm > 2.0:
            return "high_humidity"
        elif weather.humidity_pct < 40 and weather.rain_mm == 0:
            return "dry"
        return "default"

    def get_treatment(
        self, disease: str, weather: Optional[WeatherData] = None
    ) -> dict:
        protocols = self.TREATMENT_PROTOCOLS.get(
            disease, self.TREATMENT_PROTOCOLS.get("healthy", {})
        )
        if weather:
            condition = self.classify_conditions(weather)
            treatment = protocols.get(condition, protocols.get("default", "Monitor crop."))
        else:
            treatment = protocols.get("default", "Monitor crop.")

        return {
            "disease": disease,
            "treatment": treatment,
            "weather_summary": {
                "temperature": weather.temperature_c if weather else None,
                "humidity": weather.humidity_pct if weather else None,
                "condition": self.classify_conditions(weather) if weather else None,
            },
        }
