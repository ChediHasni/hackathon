import ee
ee.Authenticate()
ee.Initialize(project='zonescorer-494214')
print(ee.Image('USGS/SRTMGL1_003').getInfo())