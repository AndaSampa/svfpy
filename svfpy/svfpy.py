# import argparse
import rasterio
from rasterio import warp
import math
from scipy.spatial.distance import euclidean
from skimage.draw import line
from skimage.util import view_as_windows
import numpy as np
from rasterio.enums import Resampling

class SVF:
  def __init__(self, 
               mds_path,
               observer_height = 1.20, 
               kernel_size = 11,
               max_radius = None):
    self.mds_src = rasterio.open(mds_path)
    self.observer_height = observer_height
    self.kernel_size = self._is_kernel_size_odd(kernel_size)
    
    if max_radius is None:
      self.max_radius = max(self.mds_src.shape)
    else:
      self.max_radius = max_radius

  def _is_kernel_size_odd(self, kernel_size):
    if int(kernel_size) % 2 == 0:
      raise ValueError("Kernel Size must be odd integer!")
    return int(kernel_size)

  def external_coordinates(self):
    kernel_size = self.kernel_size
    extremes = [[0,0], [kernel_size-1,0], [kernel_size-1, kernel_size-1], [0, kernel_size-1]]
    external_coords = []
    fuse_angles = []
    start_index = 1

    for i in range(4):
        l = line(*np.roll(extremes, -i, axis=0)[0:2].flatten())
        fuse_angles.append(np.arctan2(*(np.array(l) - kernel_size // 2)[:, start_index:]))
        external_coords.append(np.array(l, dtype='uint8').T[start_index:])

    external_coords = np.concatenate(external_coords)
    fuse_angles = np.concatenate(fuse_angles)
    fuse_angles_range = np.diff(fuse_angles, append=fuse_angles[0])
    fuse_angles_range[np.where(fuse_angles_range>np.pi)] = fuse_angles_range[np.where(fuse_angles_range>np.pi)] - 2 * np.pi

    return external_coords, fuse_angles, fuse_angles_range


def distance_matrix(size=11):
  kernel_simple = np.ones((size, size), dtype='float16')
  center_point = (int(kernel_simple.shape[0]//2), int(kernel_simple.shape[1]//2))

  distances = np.zeros((size, size), dtype='float16')

  for i in range(size):
      for j in range(size):
          distances[i][j] = euclidean([0., 0.], [i - center_point[0], j - center_point[1]])
          
  return distances

def calculate(svf:SVF):

  # Levantando os parâmetros para o calculo
  pixel_max_size = math.ceil(svf.max_radius / math.floor(svf.kernel_size / 2) * max(svf.mds_src.res))
  downscale_times = math.ceil(math.log2(pixel_max_size / max(svf.mds_src.res)) + 1)

  # Abre o MDS, com PADs
  mds =  svf.mds_src.read(1).astype('float16')
  mdss, transformations = [], []

  for i in np.arange(downscale_times):
    mds, transform = warp.reproject(source=svf.mds_src.read(1),
                                    src_transform=svf.mds_src.transform,
                                    src_crs=svf.mds_src.crs,
                                    dst_crs=svf.mds_src.crs,
                                    dst_nodata=svf.mds_src.nodata,
                                    dst_resolution=tuple([r * (2 ** i) for r in svf.mds_src.res]),
                                    resampling=Resampling.max)

    # Adiciona PAD
    mds, transform = rasterio.pad(np.squeeze(mds), transform, svf.kernel_size // 2, mode='reflect')

    print('rays ..')
    windows = view_as_windows(mds, (svf.kernel_size, svf.kernel_size))

    # print('kernel ..')
    # fuse_lines = _fuse_lines(svf)
    # list_windows = windows[:, :, fuse_lines[:, :, 0], fuse_lines[:, :, 1]]

    ######################################################
    ## TODO
    ######################################################
    # np.max(windows[np.tile(lines_mask[8], (4000, 4000,1,1))].reshape(4000,4000,6), axis=2)
    #########################################################################################

    mdss.append(np.squeeze(mds))
    transformations.append(transform)

    break
  
  return windows
  # return mdss, transformations

def _fuse_lines(svf:SVF):
  # kernel_size = svf.kernel_size
  external_coords = svf.external_coordinates()[0]
  lines = []
  center_point = (int(svf.kernel_size//2), int(svf.kernel_size//2))
  
  for c in external_coords:
      lines.append(line(*center_point, *c))

  fuse_lines = np.array([np.array(l).T for l in np.array(lines)])
  
  return fuse_lines

def _rays(svf):
  kernel = view_as_windows(svf.mds, (self.kernel_size, self.kernel_size))
  return kernel
  # return kernel[:, :, self.fuse_lines()[:, :, 0], self.fuse_lines()[:, :, 1]]