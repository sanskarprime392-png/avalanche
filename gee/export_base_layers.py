"""
Export GEE-native base RASTER layers, clipped & aligned to the AOI, to Google Drive.

Handles ONLY layers GEE is good at:
  * NASADEM elevation     -> the DEM all terrain derivatives are computed from (locally)
  * WorldClim winter T & P
  * ESA WorldCover land cover
  * MODIS snow-cover duration (SCD) climatology

NOT here (computed locally in features/terrain.py from the exported DEM, because GEE can't do
flow accumulation / 2nd-derivative terrain analysis well): slope, aspect, roughness, eastness,
northerness, plan/profile curvature, TRI, TPI, VRM, valley depth, TWI, SPI.
NOT here (from vectors, features/distances.py): distance-to-road / stream / lineament.

Run in Colab:
    import ee; ee.Authenticate(); ee.Initialize(project=config.EE_PROJECT)
    %run gee/export_base_layers.py
Then monitor tasks at https://code.earthengine.google.com/tasks (or print task.status()).
"""
import ee
import config


def get_aoi():
    return ee.Geometry.Rectangle(config.AOI_BBOX)


def _export(img, name, scale):
    aoi = get_aoi()
    task = ee.batch.Export.image.toDrive(
        image=img.clip(aoi).toFloat(),
        description=f"base_{name}",
        folder=config.GEE_EXPORT_FOLDER,
        fileNamePrefix=f"base_{name}",
        region=aoi,
        scale=scale,
        crs=config.CRS,
        maxPixels=int(1e13),
    )
    task.start()
    print(f"started export: base_{name}  (scale={scale} m)")
    return task


def export_dem():
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation").rename("elevation")
    return _export(dem, "nasadem_elevation", config.TARGET_RES_M)


def export_worldclim():
    """WorldClim V1 monthly on GEE: tavg in 0.1 deg C, prec in mm. Winter = config.SEASON_MONTHS.
    (You also have WorldClim v2 downloaded — use whichever; keep one consistent for the paper.)"""
    wc = ee.ImageCollection("WORLDCLIM/V1/MONTHLY").filter(
        ee.Filter.inList("month", config.SEASON_MONTHS)
    )
    temp = wc.select("tavg").mean().multiply(0.1).rename("temp_winter")
    prec = wc.select("prec").sum().rename("precip_winter")
    t = _export(temp, "worldclim_temp_winter", 1000)
    p = _export(prec, "worldclim_precip_winter", 1000)
    return t, p


def export_worldcover():
    lc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("lulc")
    return _export(lc, "esa_worldcover", 10)


def export_scd():
    """Snow-Cover-Duration climatology from MODIS Terra (MOD10A1, 500 m).
    NDSI_Snow_Cover: 0-100 = NDSI; >100 = cloud/fill/flags. Snow day = (40 <= NDSI <= 100).
    SCD = mean over years of the per-season snow-day count. First-pass (cloudy days not gap-
    filled); upgrade later with the DLR Global SnowPack cloud-removed product if needed."""
    def season_snowdays(year):
        year = ee.Number(year)
        start = ee.Date.fromYMD(year, config.SEASON_MONTHS[0], 1)
        end = ee.Date.fromYMD(year, config.SEASON_MONTHS[-1], 1).advance(1, "month")
        col = (ee.ImageCollection("MODIS/061/MOD10A1")
               .filterDate(start, end).select("NDSI_Snow_Cover"))
        snow = col.map(lambda im: im.gte(40).And(im.lte(100)))
        return snow.sum().rename("scd").set("year", year)

    years = ee.List(config.ANALYSIS_YEARS)
    scd = ee.ImageCollection(years.map(season_snowdays)).mean()
    return _export(scd, "modis_scd", 500)


if __name__ == "__main__":
    print(f"AOI bbox: {config.AOI_BBOX} | CRS: {config.CRS}")
    export_dem()
    export_worldclim()
    export_worldcover()
    export_scd()
    print("\nAll tasks started. They run server-side; outputs land in "
          f"Drive/{config.GEE_EXPORT_FOLDER}/. Move them to data/raw/predictors/ per DATA.md.")
