from bz2 import compress
from cmath import nan
import rioxarray
import numpy as np
import os
import tempfile
from multiprocessing import Pool
from scipy import ndimage
import rasterio
from rasterio.transform import Affine
import glob
from rioxarray import merge
from scipy import ndimage
import string

class SVF:
    def __init__(self,
                mds_file,
                downscales_path = tempfile.TemporaryDirectory(),
                kernel_size_side = 2000,
                thetas = 32,
                phis = 8, 
                observer_height = 1.5,
                max_horizontal_distance = 10000,
                max_vertical_difference = 180,
                tmp_folder = None):
        
        self.mds_file = mds_file
        self.kernel_size_side = kernel_size_side
        self.downscales_path = downscales_path
        self.xmds = rioxarray.open_rasterio(mds_file)
        self.xmds0 = self.xmds[0]
        self.profile = rasterio.open(mds_file).profile
        self.resolution = self.xmds.rio.resolution()[0]
        self.max_vertical_difference = max_vertical_difference
        self.delta_theta = 2*np.pi/thetas
        self.phis_angles = np.arcsin(np.arange(1/phis/2, 1, 1/phis))
        self.tangents = 1/np.tan(self.phis_angles) 
        self.pad_kernel = np.ceil(self.kernel_size_side / 2 * np.sqrt(2)) - self.kernel_size_side / 2
        self.horizontal_distance = max(self.downscales() * self.resolution * self.kernel_size_side)
        self.resolutions = np.unique(self.downscales() * self.resolution)
        self.max_horizontal_distance = max_horizontal_distance
        self.thetas = thetas # Must be divisible by 4
        self.tmp_folder = self._set_tmp_folder(tmp_folder)
        self.observer_height = observer_height

    def downscales(self):
        downscales = np.floor(self.tangents[::-1] * (1/self.resolution))
        downscales = np.where(downscales == 0, 1, downscales)
        return downscales

    def get_downscale(self,
                    resolution, 
                    force=False):
        if resolution == self.resolution:
            mds = self.xmds
        else:            
            mds_downscale = self.downscales_path + os.path.basename(self.mds_file).split('.')[0] + f'_{int(resolution * 100)}cm_' + '.tif'
            if not os.path.exists(mds_downscale) or force:
                os.system(f'gdal_translate -tr {resolution} {resolution} -r mode -of GTiff {self.mds_file} {mds_downscale}')
            mds = rioxarray.open_rasterio(mds_downscale)
        return mds

    def prepare_downscales(self, force=False) -> list:
        d_scales = []
        for i in np.unique(self.downscales()):
            print(f'Preparing Downscale, resolution of {i * self.resolution}')
            d_scales.append(self.get_downscale(i * self.resolution, force=force))
        return d_scales

    def pad_max(self) -> np.array:
        pad_tangent = (self.tangents[::-1] * self.max_vertical_difference) / self.downscales()
        pad_max = np.ceil(((self.kernel_size_side / 2) + pad_tangent) * np.sqrt(2) - (self.kernel_size_side / 2) + pad_tangent)
        # Adding Max Distance
        diff_to_horizon_max =  self.max_horizontal_distance / self.downscales()[-1] / self.resolution
        if diff_to_horizon_max > pad_tangent[-1]:
            pad_max[-1] = pad_max[-1] + diff_to_horizon_max - pad_tangent[-1]
        return np.int16(pad_max)

    def pad_max_by_resolution(self):
        dict = {}
        for a,b in zip (self.downscales() * self.resolution, self.pad_max()):
            dict[a] = b
        return dict

    def working_kernel(self, row, col, resolution):
        # REtorna o kernel de trabalho com PAD
        rows, cols = self.kernels(resolution)
        assert row <= rows, f"Row must be <= than {rows}"
        assert col <= cols, f"Column must be <= than {cols}"

        mds = self.get_downscale(resolution)

        col_s = col * self.kernel_size_side 
        col_f = (col + 1) * (self.kernel_size_side) 
        row_s = row * self.kernel_size_side 
        row_f = (row + 1) * self.kernel_size_side 
        
        pad_pixels = self.pad_max_by_resolution()[resolution]
        
        col_s_padded = col_s - pad_pixels
        col_f_padded = col_f + pad_pixels
        row_s_padded = row_s - pad_pixels
        row_f_padded = row_f + pad_pixels
        
        pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0

        if col_s_padded < 0:
            pad_left = int(abs(col_s_padded))
            col_s_padded = 0

        if row_s_padded < 0:
            pad_top = int(abs(row_s_padded))
            row_s_padded = 0
        
        if col_f_padded >= mds.shape[2]:
            pad_right = int(col_f_padded - mds.shape[2])
            col_f_padded = mds.shape[2]# - 1

        if row_f_padded >= mds.shape[1]:
            pad_bottom = int(row_f_padded - mds.shape[1])
            row_f_padded = mds.shape[1]# - 1
        
        k_slice = (pad_left, pad_right), (pad_top, pad_bottom), (row_s_padded), (row_f_padded), (col_s_padded), (col_f_padded)

        w_kernel_padded = mds[0, int(row_s_padded):int(row_f_padded), int(col_s_padded):int(col_f_padded)]
        w_kernel_padded = np.pad(w_kernel_padded, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=nan)
        
        return k_slice, w_kernel_padded

    def kernels(self, resolution):
        mds = self.get_downscale(resolution)

        rows = mds.shape[1] // self.kernel_size_side
        cols =  mds.shape[2] // self.kernel_size_side 

        return rows, cols

    def svf(self):
        for res in self.resolutions:
            row, col = self.kernels(res)
            for row in range(row + 1):
                for col in range(col + 1):
                    if not os.path.exists(f'{self.tmp_folder}{row}_{col}_{res}_temptest.tiff'):
                        self.svf_kernel(row, col, res)

            print(f'Agregando {res}')
            self._agg_scale(res)
        # self._agg_all()    
        return None

    def svf_kernel(self, row, col, res):

        print(row, col, res)
        
        mds, svf = self._calc_svf(row, col, res)
        # sk, wk = self.working_kernel(row, col, res)
        # write file
        if not np.all(svf == 0.): 
            x = self.xmds.rio.bounds()[0]
            y = self.xmds.rio.bounds()[-1]
            transform = Affine.translation(x + col * self.kernel_size_side * res, y - (row) * self.kernel_size_side * res) * Affine.scale(res, -res)
            profile = self.profile
            profile.update(
                transform=transform,
                height=svf.shape[0],
                width=svf.shape[1]
                # height=2500,
                # width=2500
            )

            profile.update(compress='lzw')

            ## TODO
            ## Convert to RioXArray
            ## https://github.com/corteva/rioxarray/discussions/430
            with rasterio.open(f'{self.tmp_folder}{row}_{col}_{res}_temptest.tiff', 'w', **profile) as svf_part_out:
                svf_part_out.write(svf, 1)

        return svf


    def _agg_scale(self, resolution):
        vrt_file = f'{self.tmp_folder}all_{resolution}_temptest.vrt'
        passes = len(np.where(self.downscales() * self.resolution == resolution)[0])
        max_value = 1 + (passes * self.thetas)
        bounds = self.xmds.rio.bounds()
        projwin = ' '.join([str(bounds[0]), str(bounds[3]), str(bounds[2]), str(bounds[1])])
        
        os.system(f'gdalbuildvrt -srcnodata -9999 -vrtnodata {max_value} {vrt_file} {self.tmp_folder}*_{resolution}_temptest.tiff')
        os.system(f'gdal_translate -tr {self.resolution} {self.resolution} -r average -a_nodata {max_value} -projwin {projwin} {vrt_file} {self.tmp_folder}all_{resolution}_upscaled_{self.resolution}.tiff -co "COMPRESS=ZSTD" -co "PREDICTOR=3" -co "BIGTIFF=YES"')
        
        return None

    def _agg_all(self):
        files = [f'{self.tmp_folder}all_{resolution}_upscaled_0.5.tiff' for resolution  in self.resolutions]
        letters = [letter for letter in string.ascii_uppercase[0:len(files)]]
        files_argument = " ".join([f'-{l} {i}' for l, i in zip(letters, files)])
        calc = '+'.join(letters)
        command_line = f'gdal_calc.py {files_argument} --outfile=svf_all.tiff --calc="{calc}" --quiet --overwrite --hideNoData --NoDataValue=-9999 --co="COMPRESS=ZSTD" --co="PREDICTOR=3" --co="BIGTIFF=YES" --extent=intersect'
        
        # svfs = glob.glob(f'{self.tmp_folder}all_*_temptest.tiff')
        # rasters = []
        # for s in svfs:
        #     rasters.append(rioxarray.open_rasterio(s, engine="rasterio", chunks=True, masked=True))
        # xmds = merge.merge_arrays(rasters, res=(0.5, 0.5), method='sum')
        # xmds.rio.to_raster(f'{self.tmp_folder}all_temptest.tiff', compress='lzw')
        return command_line

    def calc_svf_point(self, row:int, col:int):
        # mds = self.xmds
        row_idx, col_idx = np.indices(self.xmds0.shape)
        row_idx -= row
        col_idx -= col
        theta_angles = np.arctan2(row_idx, col_idx) + np.pi
        distances = np.hypot(row_idx, col_idx)
        # deltas =  self.xmds0 - (self.xmds0[0, row, col] + self.observer_height)
        deltas =  self.xmds0 - (self.xmds0[row, col])
        phi_angles = np.nan_to_num(np.arctan2(deltas, distances))
        phi_angles[phi_angles < 0] = 0.
        svf = 0.
        angles = 0
        for a in np.linspace(0, 2*np.pi, self.thetas, endpoint=False):
            mask = np.logical_and((theta_angles >= a), (theta_angles < (a + self.delta_theta)))
            if len(phi_angles[mask]) > 0:
                angles += 1
                svf += (1. - np.sin(np.max(phi_angles[mask]))) #/ self.thetas
        return svf / angles
        # return mask 


    def _calc_svf(self, row, col, resolution):

        mds, wk = self.working_kernel(row, col, resolution)

        wk = np.nan_to_num(wk)

        tangs = self.tangents[::-1][np.where(self.downscales() * self.resolution == resolution)[0]]
        pad = self.pad_max_by_resolution()[resolution]
        svf = np.ones(wk[pad:-pad, pad:-pad].shape, dtype='int16')
        # print(len(tangs))

        # Test if all values is nan
        if np.all(wk == 0.):
            return None, wk

        for i in np.linspace(0, np.pi/2, self.thetas//4, endpoint=False):

            # PErformance ISSUE
            if wk.shape[0] == wk.shape[1]:
                mds_r = rotate_2d(wk, i)
            else:
                mds_r = ndimage.rotate(wk, np.rad2deg(i), reshape=False)

            # print(np.rad2deg(i))

            svf_part = []
            # for q in range(1):
            # for q, t in np.array([[r, t] for r in range(4) for t in tangs]):
            #     svf_part.append(_calc_svf_quadrant(mds_r, q, t, resolution))
            
            ## MULTIPROCESSING OPTION
            p_loop = np.array([[mds_r, r, t, resolution] for r in range(4) for t in tangs], dtype=object)
            with Pool(12) as p:
                svf_part = p.starmap(_calc_svf_quadrant, zip(p_loop[:, 0], p_loop[:, 1], p_loop[:, 2], p_loop[:, 3]))

            svf += rotate_2d(np.sum(np.array(svf_part), axis=0), -i)[pad:-pad, pad:-pad]
        return mds, svf

    def _set_tmp_folder(self, tmp_folder):
        if tmp_folder is None:
            # TODO - Set Env Temp Folder
            return '../tmp/'
        else:
            return tmp_folder

def _calc_svf_quadrant(mds, quadrant, tangent, resolution):
    indices = np.indices(mds.shape)
    # print(indices[1])
    # MAking Projection
    projection = np.rot90(mds, k=quadrant) * tangent + resolution * indices[1]
    # Accumulated Projection
    projection_acc = np.maximum.accumulate(projection, axis=1)
    # Calculating visibility
    sky_is_visible = np.less_equal(projection_acc, projection)
    # quadrant back
    return np.rot90(sky_is_visible, k=-quadrant)

def rotate_2d(matrix, angle):
    shape = matrix.shape
    
    y, x = np.indices(shape,  dtype='float32')
    x, y = x - shape[0]//2, y[::-1] - shape[0]//2

    xr = np.int32(np.cos(angle) * x - np.sin(angle) * y[::-1] + shape[0]//2)
    yr = np.int32(np.sin(angle) * x + np.cos(angle) * y[::-1] + shape[0]//2)
    

    xr = np.where(xr < 0, 0, xr)
    xr = np.where(xr > shape[0]-1, shape[0]-1, xr)
    yr = np.where(yr < 0, 0, yr)
    yr = np.where(yr > shape[1]-1, shape[1]-1, yr)

    return matrix[yr, xr]

def zoom2d(matrix, times:int):
  shape  = matrix.shape
  dims = len(shape)

  matrix_zomed = \
  np.tile(
    np.expand_dims(
      np.tile(
        np.expand_dims(matrix, axis=2),
        tuple(np.concatenate([np.array([1,1, times]), np.ones(dims-2, dtype='int')]))
      ).reshape(
        tuple(np.array(shape) * np.concatenate([np.array([1,times]), np.ones(dims-2, dtype='int')]))
      ),
      axis=1
    ),
    tuple(np.concatenate([np.array([1,times]), np.ones(dims-1, dtype='int')]))
  ).reshape(
    tuple(np.array(shape) * np.concatenate([np.array([times,times]), np.ones(dims-2, dtype='int')]))
  )
  
  return matrix_zomed

    