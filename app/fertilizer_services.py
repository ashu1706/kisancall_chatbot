from app.fertilizers_api import get_field_health, get_soil_moisture, recommend_fertilizer

# Temporary polygon_id (replace with real one after creating polygon on AgroMonitoring API)
TEST_POLYGON_ID = "65d5e887d4e9ba0007e96e2f"

def get_fertilizer_recommendation(crop: str, lat: float, lon: float) -> str:
    """
    Fetch field data (NDVI + soil moisture) and return fertilizer recommendation.
    """
    try:
        ndvi_data = get_field_health(TEST_POLYGON_ID)
        soil_data = get_soil_moisture(TEST_POLYGON_ID)

        if ndvi_data["status"] != "success" or soil_data["status"] != "success":
            return "❌ Could not fetch satellite data for your farm. Please try again later."

        # Extract values
        ndvi_value = ndvi_data["data"][-1]["mean"] if ndvi_data["data"] else 0.4
        soil_moisture = soil_data["data"].get("moisture", 0.3)

        recommendation = recommend_fertilizer(ndvi_value, soil_moisture)

        return (
            f"📊 Fertilizer Recommendation for {crop}:\n"
            f"- NDVI (crop health index): {ndvi_value:.2f}\n"
            f"- Soil Moisture: {soil_moisture:.2f}\n\n"
            f"💡 {recommendation}"
        )
    except Exception as e:
        return f"⚠️ Error while generating fertilizer recommendation: {str(e)}"
