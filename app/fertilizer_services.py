from app.fertilizers_api import get_field_health, get_soil_moisture, recommend_fertilizer
from app.user_polygons import get_polygon

def get_fertilizer_recommendation(user_id: str, crop: str, lat: float, lon: float) -> str:
    """
    Fetch field data (NDVI + soil moisture) and return fertilizer recommendation.
    """
    try:
        polygon_id = get_polygon(user_id)
        if not polygon_id:
            return "⚠️ No farm polygon linked for this user. Please register your farm first."

        # Fetch NDVI & soil data
        ndvi_resp = get_field_health(polygon_id)
        soil_resp = get_soil_moisture(polygon_id)

        if ndvi_resp["status"] != "success" or soil_resp["status"] != "success":
            return "Could not fetch satellite data for your farm. Please try again later."

        # NDVI → take last scene if available
        ndvi_list = ndvi_resp["data"]
        if isinstance(ndvi_list, list) and len(ndvi_list) > 0:

            ndvi_value = ndvi_list[-1]["data"].get("mean", 0.4)
        else:
            ndvi_value = 0.4  # fallback

        # Soil moisture
        soil_data = soil_resp["data"]
        soil_moisture = soil_data.get("moisture", 0.3) if isinstance(soil_data, dict) else 0.3

        # Get recommendation
        recommendation = recommend_fertilizer(ndvi_value, soil_moisture)

        return (
            f"📊 Fertilizer Recommendation for {crop}:\n"
            f"- NDVI (crop health index): {ndvi_value:.2f}\n"
            f"- Soil Moisture: {soil_moisture:.2f}\n\n"
            f"💡 {recommendation}"
        )

    except Exception as e:
        return f"⚠️ Error while generating fertilizer recommendation: {str(e)}"
