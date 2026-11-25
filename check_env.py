import importlib
mods = ['django','torch','h3','geopandas','torch_geometric','osmnx','earthengine','cdsapi','requests','numpy','pandas','shapely','xarray']
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, '__version__', 'ok')
        print(f'{m}: OK ({ver})')
    except Exception as e:
        print(f'{m}: FAIL ({type(e).__name__}: {e})')
