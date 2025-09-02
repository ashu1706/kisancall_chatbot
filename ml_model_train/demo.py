import joblib
import os
import pandas as pd
import numpy as np


# --- All Helper functions are placed here at the beginning ---

def get_example_data(location, crop, growth_stage):
    """
    Simulates fetching real-time and soil data for a specific scenario.
    In a real app, this would come from APIs and databases.
    """
    if location == "Punjab" and crop == "Wheat":
        # Example data for wheat in Punjab (based on your dataset)
        return {
            'weather_data': {
                'daily': {'temperature_2m_max': [25.5], 'temperature_2m_min': [12.1], 'precipitation_sum': [0]},
                'hourly': {'relative_humidity_2m': [45.0]},
                'recent': {'rainfall': [0] * 24}
            },
            'soil_data': {
                'Soil_pH': 7.2,
                'Soil_Nitrogen_kg_ha': 150.5,
                'Soil_Phosphorus_kg_ha': 45.3,
                'Soil_Potassium_kg_ha': 280.1,
                'NPK_Ratio': 0.5,
                'Organic_Carbon_%': 1.5
            },
            'user_context': {
                'crop': crop,
                'location_name': location,
                'sowing_season': 'Rabi',
                'soil_type': 'Loamy',
                'growth_stage': growth_stage
            }
        }
    else:
        return None


def prepare_irrigation_input(data):
    """
    Prepares the input data for the irrigation model, ensuring all columns are present.
    """
    # Combine the user context and weather/soil data into a single dictionary
    input_features = {
        'State': [data['user_context']['location_name']],
        'Crop': [data['user_context']['crop']],
        'Growth_Stage': [data['user_context']['growth_stage']],
        'Soil_Type': [data['user_context']['soil_type']],
        'Rainfall_mm': [data['weather_data']['daily']['precipitation_sum'][0]],
        'Temp_Max_C': [data['weather_data']['daily']['temperature_2m_max'][0]],
        'Temp_Min_C': [data['weather_data']['daily']['temperature_2m_min'][0]],
        'Humidity_Percent': [data['weather_data']['hourly']['relative_humidity_2m'][0]],
        'Soil_Moisture_Percent': [0.45]  # Placeholder value for demonstration
    }

    # Create the DataFrame from the combined data
    return pd.DataFrame(input_features)


def prepare_fertilizer_input(data):
    """
    Prepares the input data for the fertilizer model, ensuring all columns are present.
    """
    # Combine the user context and weather/soil data into a single dictionary
    input_features = {
        'State': [data['user_context']['location_name']],
        'Crop': [data['user_context']['crop']],
        'Growth_Stage': [data['user_context']['growth_stage']],
        'Soil_Type': [data['user_context']['soil_type']],
        'Rainfall_mm': [data['weather_data']['daily']['precipitation_sum'][0]],
        'Soil_pH': [data['soil_data']['Soil_pH']],
        'Soil_Nitrogen_kg_ha': [data['soil_data']['Soil_Nitrogen_kg_ha']],
        'Soil_Phosphorus_kg_ha': [data['soil_data']['Soil_Phosphorus_kg_ha']],
        'Soil_Potassium_kg_ha': [data['soil_data']['Soil_Potassium_kg_ha']],
        'Previous_Crop': ['Maize'],  # Placeholder for the previous crop
        'NPK_Ratio': [data['soil_data']['NPK_Ratio']]
    }

    # Create the DataFrame from the combined data
    return pd.DataFrame(input_features)


def prepare_pest_input(data):
    """
    Prepares the input data for the pest model, ensuring all columns are present.
    """
    input_features = {
        'Crop': [data['user_context']['crop']],
        'Location': [data['user_context']['location_name']],
        'Sowing_Season': [data['user_context']['sowing_season']],
        'Temperature_Max_C': [data['weather_data']['daily']['temperature_2m_max'][0]],
        'Humidity_Avg_%': [data['weather_data']['hourly']['relative_humidity_2m'][0]],
        'Rainfall_mm': [data['weather_data']['daily']['precipitation_sum'][0]]
    }
    return pd.DataFrame(input_features)


# --- Main Demonstration Logic ---

def run_demonstration(location, crop, growth_stage):
    print("--- Running Kisan Chatbot Demonstration ---")
    print(f"Scenario: {location} - {crop} crop at {growth_stage} stage.")

    # Get sample data
    data = get_example_data(location, crop, growth_stage)
    if not data:
        print("Error: No sample data for this scenario.")
        return

    # Load ML models
    try:
        # Since models are in the same folder, just use the filename
        irrigation_model = joblib.load('irrigation_model.joblib')
        preprocessor_irrigation = joblib.load('data_preprocessor.joblib')
        fertilizer_type_model = joblib.load('fertilizer_type_model1.joblib')
        fertilizer_quantity_model = joblib.load('fertilizer_quantity_model1.joblib')
        preprocessor_fertilizer = joblib.load('data_preprocessor1.joblib')
        pest_model = joblib.load('pest_advisory_model.joblib')
        preprocessor_pest = joblib.load('data_preprocessor_pest.joblib')
        print("✅ All models loaded successfully!")
    except Exception as e:
        print(f" Failed to load models: {e}")
        return

    # Prepare data for each model
    irrigation_df = prepare_irrigation_input(data)
    fertilizer_df = prepare_fertilizer_input(data)
    pest_df = prepare_pest_input(data)

    # Make predictions
    irrigation_needed = irrigation_model.predict(preprocessor_irrigation.transform(irrigation_df))[0]
    fertilizer_type = fertilizer_type_model.predict(preprocessor_fertilizer.transform(fertilizer_df))[0]
    fertilizer_quantity = fertilizer_quantity_model.predict(preprocessor_fertilizer.transform(fertilizer_df))[0]
    pest_risk = pest_model.predict(preprocessor_pest.transform(pest_df))[0]

    # --- Print the Final Advisory ---
    print("\n--- Generated Advisory ---")
    print(f"📍 **Location:** {location}")
    print(f"🌾 **Crop:** {crop}")
    print(f"🌱 **Growth Stage:** {growth_stage}")
    print("\n")
    print(
        f" **Irrigation Advisory:** Based on current weather, your crop needs approximately {irrigation_needed:.2f} mm of water.")
    print(
        f" **Fertilizer Recommendation:** Your soil requires **{fertilizer_type}** at a dosage of approximately {fertilizer_quantity:.2f} kg/ha.")
    print(
        f" **Pest & Disease Risk:** The current risk is **{pest_risk}**. It is recommended to monitor your crop closely.")
    print("\n----------------------------------------")


if __name__ == '__main__':
    run_demonstration("Punjab", "Wheat", "Tillering")
