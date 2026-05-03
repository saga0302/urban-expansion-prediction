import ee
import geemap
import os

# Initialize Earth Engine
ee.Initialize(project='roboflow-489117')

# Output directory
output_dir = 'data/raw'
os.makedirs(output_dir, exist_ok=True)

# Asia bounding box
asia = ee.Geometry.Rectangle([25.0, 1.0, 180.0, 81.0])

print("Downloading Population Density (GPW v4)...")
population = ee.Image("CIESIN/GPWv411/GPW_Population_Density/gpw_v4_population_density_rev11_2020_30_sec") \
    .select('population_density') \
    .clip(asia)
geemap.ee_export_image(
    population,
    filename=f'{output_dir}/population_density.tif',
    scale=10000,
    region=asia,
    crs='EPSG:4326'
)

print("Downloading Nighttime Lights (VIIRS 2020)...")
lights = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG") \
    .filterDate('2020-01-01', '2020-12-31') \
    .mean() \
    .select('avg_rad') \
    .clip(asia)
geemap.ee_export_image(
    lights,
    filename=f'{output_dir}/nighttime_lights.tif',
    scale=10000,
    region=asia,
    crs='EPSG:4326'
)

print("Downloading Elevation (SRTM)...")
elevation = ee.Image("USGS/SRTMGL1_003") \
    .select('elevation') \
    .clip(asia)
geemap.ee_export_image(
    elevation,
    filename=f'{output_dir}/elevation.tif',
    scale=10000,
    region=asia,
    crs='EPSG:4326'
)

print("Downloading NDVI Vegetation (MODIS 2020)...")
ndvi = ee.ImageCollection("MODIS/061/MOD13A3") \
    .filterDate('2020-01-01', '2020-12-31') \
    .mean() \
    .select('NDVI') \
    .clip(asia)
geemap.ee_export_image(
    ndvi,
    filename=f'{output_dir}/ndvi.tif',
    scale=10000,
    region=asia,
    crs='EPSG:4326'
)

print("All done! Check data/raw/ folder")