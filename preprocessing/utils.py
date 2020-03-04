import numpy as np
import gdal
import osr
from pathlib import Path

# reading in geotiff file as numpy array
def read_tif(file: Path):

    if not file.exists():
        raise FileNotFoundError(f'File {file} not found')

    ds = gdal.Open(str(file))

    geotransform = ds.GetGeoTransform()

    proj = osr.SpatialReference(wkt=ds.GetProjection())
    epsg = int(proj.GetAttrValue('AUTHORITY' ,1))

    xy_shape = np.array(ds.GetRasterBand(1).ReadAsArray()).shape

    # get number of bands in raster file
    n_bands = ds.RasterCount

    # initialize a data cube
    xyz_shape = xy_shape + (n_bands,)
    data_cube = np.ndarray(xyz_shape)

    # fill it with bands
    for i in range(1, n_bands+1):
        data_cube[: ,: , i -1] = np.array(ds.GetRasterBand(i).ReadAsArray())

    ds = None
    return data_cube, geotransform, epsg
    # end of read in datacube function


# writing an array to a geo tiff file
def write_tif(arr, geotransform, epsg, save_dir: Path, fname: str, dtype=gdal.GDT_Float32):

    if not save_dir.exists():
        save_dir.mkdir()
    file = save_dir / f'{fname}.tif'

    n_rows, n_cols = arr.shape[:2]
    n_bands = arr.shape[2] if len(arr.shape) > 2 else 1

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(file), n_rows, n_cols, n_bands, dtype)

    # setting coordinate reference system
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())

    # write data to file
    if n_bands == 1:
        arr = arr if len(arr.shape) == 2 else arr[:, :, 0]
        ds.GetRasterBand(1).WriteArray(arr)
    else:
        for i in range(n_bands):
            ds.GetRasterBand(i + 1).WriteArray(arr[:, :, i])

    ds.FlushCache()  # saves to disk
    del driver
    del ds

