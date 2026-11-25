import ee
import h3

# --- Auth ---
credentials = ee.ServiceAccountCredentials(
    email='zonescorer@zonescorer-494214.iam.gserviceaccount.com',
    key_file=r"C:\Users\User\Desktop\hackathon\hackathon\zonescorer-494214-6afbb2def406.json"
)
ee.Initialize(credentials, project='zonescorer-494214')
print("✅ Authenticated")

# --- Single H3 cell over Tunis ---
cell = h3.latlng_to_cell(36.81, 10.18, 7)  # one H3 cell at Tunis
print(f"✅ H3 cell: {cell}")

# --- Build geometry ---
boundary = h3.cell_to_boundary(cell)
coords = [[lng, lat] for lat, lng in boundary]
coords.append(coords[0])
poly = ee.Geometry.Polygon(coords)

# --- Fetch NDVI ---
s2 = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(poly)
    .filterDate('2024-01-01', '2024-06-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .median()
)

ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')

result = ndvi.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=poly,
    scale=10
).getInfo()

print(f"✅ NDVI result: {result}")
print(f"✅ NDVI value: {result.get('NDVI')}")

