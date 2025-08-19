# fertilizers_api.py
import requests
import os

# Example: AgroMonitoring API base URL (you can replace with Farmonaut or Agrio later)
AGROMONITORING_API_KEY = os.getenv("AGROMONITORING_API_KEY", "e4d776f415123cdeb26e752235fdf5e2")
AGROMONITORING_BASE_URL = "http://api.agromonitoring.com/agro/1.0"

def get_field_health(polygon_id: str):
    """
    Fetch NDVI and crop health data for a given field polygon.
    """
    url = f"{AGROMONITORING_BASE_URL}/ndvi?polyid={polygon_id}&appid={AGROMONITORING_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def get_soil_moisture(polygon_id: str):
    """
    Fetch soil moisture data for a given polygon.
    """
    url = f"{AGROMONITORING_BASE_URL}/soil?polyid={polygon_id}&appid={AGROMONITORING_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def recommend_fertilizer(ndvi_value: float, soil_moisture: float):
    """
    Very basic fertilizer recommendation logic (placeholder).
    - Low NDVI & Low Soil Moisture → NPK boost
    - Low NDVI & Good Soil Moisture → Nitrogen boost
    - High NDVI & Low Soil Moisture → Potassium for stress resistance
    """
    if ndvi_value < 0.3 and soil_moisture < 0.2:
        return "Apply balanced NPK fertilizer (20:20:20) to improve crop growth."
    elif ndvi_value < 0.3:
        return "Apply Urea (Nitrogen-rich fertilizer) to boost vegetative growth."
    elif soil_moisture < 0.2:
        return "Apply Potassium-based fertilizer to improve stress resistance."
    else:
        return "Crop health looks good. Maintain current fertilizer schedule."
