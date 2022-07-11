# import argparse
import rasterio
import math
from scipy.spatial.distance import euclidean
from skimage.draw import line
from skimage.util import view_as_windows
import numpy as np

class SVF:
  def __init__(self, 
               mds_path,
               observer_height = 1.20, 
               kernel_size = 11,
               max_radius = None):
    self.mds_src = rasterio.open(mds_path)
    self.mds = self.mds_src.read(1).astype('float16')
    self.observer_height = observer_height
    self.kernel_size = self._is_kernel_size_odd(kernel_size)
    
    if max_radius is None:
      self.max_radius = max(self.mds_src.shape)
      # self.max_radius = math.ceil(math.sqrt(self.mds_src.shape[0] ** 2 + self.mds_src.shape[-1] ** 2))
    else:
      self.max_radius = max_radius

    self.pixel_max_size = math.ceil(self.max_radius / math.floor(self.kernel_size / 2) * max(self.mds_src.res))
    self.downscale_times = math.ceil(math.log2(self.pixel_max_size / max(self.mds_src.res)) + 1)

    self.view_as_windows = view_as_windows(self.mds, (self.kernel_size, self.kernel_size))

  def _is_kernel_size_odd(self, kernel_size):
    if int(kernel_size) % 2 == 0:
      raise ValueError("Kernel Size must be odd integer!")
    return int(kernel_size)

  def rays(self):
    kernel = self.view_as_windows
    return kernel[:, :, self.fuse_lines()[:, :, 0], self.fuse_lines()[:, :, 1]]

  def external_coordinates(self):
    kernel_size = self.kernel_size
    extremes = [[0,0], [kernel_size-1,0], [kernel_size-1, kernel_size-1], [0, kernel_size-1]]
    external_coords = []
    fuse_angles = []
    start_index = 1

    for i in range(4):
        l = line(*np.roll(extremes, -i, axis=0)[0:2].flatten())
    #     if i == 3:
    #         start_index = 0
        fuse_angles.append(np.arctan2(*(np.array(l) - kernel_size // 2)[:, start_index:]))
        external_coords.append(np.array(l, dtype='uint8').T[start_index:])
    #     start_index = 1

    external_coords = np.concatenate(external_coords)
    fuse_angles = np.concatenate(fuse_angles)
    fuse_angles_range = np.diff(fuse_angles, append=fuse_angles[0])
    fuse_angles_range[np.where(fuse_angles_range>np.pi)] = fuse_angles_range[np.where(fuse_angles_range>np.pi)] - 2 * np.pi

    return external_coords, fuse_angles, fuse_angles_range

  def fuse_lines(self):
    kernel_size = self.kernel_size
    external_coords = self.external_coordinates()[0]
    lines = []
    center_point = (int(kernel_size//2), int(kernel_size//2))
    for c in external_coords:
        lines.append(line(*center_point, *c))
    #     print(*center_point, *c)
    #     break

    fuse_lines = np.array([np.array(l).T for l in np.array(lines)])

    return fuse_lines

def distance_matrix(size=11):
  kernel_simple = np.ones((size, size), dtype='float16')
  center_point = (int(kernel_simple.shape[0]//2), int(kernel_simple.shape[1]//2))

  distances = np.zeros((size, size), dtype='float16')

  for i in range(size):
      for j in range(size):
          distances[i][j] = euclidean([0., 0.], [i - center_point[0], j - center_point[1]])
          
  return distances




