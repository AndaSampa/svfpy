import rioxarray
import numpy as np
import os
import tempfile

class SVF:
    def __init__(self,
                mds_file,
                downscales_path = tempfile.TemporaryDirectory(),
                kernel_size_side = 2000,
                thetas = 32,
                phis = 8, 
                observer_height = 1.5,
                max_horizontal_distance = 10000,
                max_vertical_difference = 180):
        
        self.mds_file = mds_file
        self.kernel_size_side = kernel_size_side
        self.downscales_path = downscales_path
        self.xmds = rioxarray.open_rasterio(mds_file)
        self.resolution = self.xmds.rio.resolution()[0]
        self.shape = np.int16(np.ceil(np.array(self.xmds.rio.shape) * self.resolution / self.kernel_size_side))
        self.max_vertical_difference = max_vertical_difference
        self.delta_theta = 2*np.pi/thetas
        self.phis_angles = np.arcsin(np.arange(1/phis/2 + 1/phis, 1, 1/phis))
        self.tangents = 1/np.tan(self.phis_angles) 
        self.pad = np.ceil(self.kernel_size_side / 2 * np.sqrt(2)) - self.kernel_size_side / 2

    ## TODO
    # Assert PAD size and square rotation
    # np.ceil(svf.pad_max() / (svf.downscales() * svf.resolution)), svf.downscales() * svf.resolution

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
                os.system(f'gdalwarp -tr {resolution} {resolution} -r average -of GTiff -overwrite {self.mds_file} {mds_downscale}')
            mds = rioxarray.open_rasterio(mds_downscale)
        return mds

    def prepare_downscales(self, force=False) -> list:
        d_scales = []
        for i in np.unique(self.downscales()):
            print(f'Preparing Downscale, resolution of {i * self.resolution}')
            d_scales.append(self.get_downscale(i * self.resolution, force=force))
        return d_scales

    def pad_max(self) -> np.array:
        pad_max = np.ceil((self.tangents[::-1] * self.max_vertical_difference))
        return pad_max

    def working_kernel(self, col, row, resolution):
        # REtorna o kernel de trabalho com PAD
        assert col <= tuple(self.shape)[0] - 1, f"Column must be <= than {tuple(self.shape)[0] - 1}"
        assert row <= tuple(self.shape)[1] - 1, f"Row must be <= than {tuple(self.shape)[1] - 1}"

        mds = self.get_downscale(resolution)

        col_s = col * self.kernel_size_side / resolution
        col_f = (col + 1) * (self.kernel_size_side) / resolution
        
        if col_f > mds.shape[1]:
            col_f = mds.shape[1]

        row_s = row * self.kernel_size_side  / resolution
        row_f = (row + 1) * self.kernel_size_side  / resolution

        if row_f > mds.shape[2]:
            row_f = mds.shape[2]
        
        # w_kernel = ((int(col_s),int(col_f)), (int(row_s),int(row_f)))

        pad_pixels = np.int16(np.ceil(self.pad / resolution))
        # print(pad_pixels)
        
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
        
        if col_f_padded > mds.shape[1]:
            pad_right = int(col_f_padded - mds.shape[1])
            col_f_padded = mds.shape[1]

        if row_f_padded > mds.shape[2]:
            pad_bottom = int(row_f_padded - mds.shape[2])
            row_f_padded = mds.shape[2]
        
        print((pad_left, pad_right), (pad_top, pad_bottom))

        w_kernel_padded = mds[0, int(col_s_padded):int(col_f_padded), int(row_s_padded):int(row_f_padded)]
        w_kernel_padded = np.pad(w_kernel_padded, ((pad_left, pad_right), (pad_top, pad_bottom)), mode='constant', constant_values=np.nan)
        
        return pad_pixels, w_kernel_padded
        # mds[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels]


def rotate_2d(matrix, angle):
    shape = matrix.shape
    
    y, x = np.indices(shape,  dtype='float32')
    x, y = x - shape[0]//2, y[::-1] - shape[0]//2

    xr = np.int32(np.cos(angle) * x - np.sin(angle) * y[::-1] + shape[0]//2)
    yr = np.int32(np.sin(angle) * x + np.cos(angle) * y[::-1] + shape[0]//2)
    

    xr = np.where(xr < 0, 0, xr)
    xr = np.where(xr > shape[0]-1, shape[0]-1, xr)
    yr = np.where(yr < 0, 0, yr)
    yr = np.where(yr > shape[0]-1, shape[0]-1, yr)

    return matrix[yr, xr]