import requests
import json

API_KEY = "e4d776f415123cdeb26e752235fdf5e2"

# Example polygon (square around Bangalore – replace with your farm's coordinates)
polygon = {
    "name": "Wheat Farm Test",
    "geo_json": {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [77.5946, 12.9716],
                [77.5956, 12.9716],
                [77.5956, 12.9726],
                [77.5946, 12.9726],
                [77.5946, 12.9716]
            ]]
        }
    }
}

url = f"http://api.agromonitoring.com/agro/1.0/polygons?appid={API_KEY}"
response = requests.post(url, json=polygon)

print("Status:", response.status_code)
print("Response:", json.dumps(response.json(), indent=2))
