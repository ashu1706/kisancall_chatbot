# app/user_polygons.py

# Temporary storage (replace with DB later)
USER_POLYGONS = {}

def set_polygon(user_id: str, polygon_id: str):
    USER_POLYGONS[user_id] = polygon_id
    return USER_POLYGONS[user_id]

def get_polygon(user_id: str) -> str:
    return USER_POLYGONS.get(user_id)
