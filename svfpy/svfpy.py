# import argparse
import rasterio
from rasterio import warp
import math
from scipy.spatial.distance import euclidean
from skimage.draw import line
from skimage.util import view_as_windows
import numpy as np
from rasterio.enums import Resampling
import zarr

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

def lines_mask(svf:SVF):
  lines_mask = np.zeros((_fuse_lines(svf).shape[0], svf.kernel_size, svf.kernel_size), dtype='bool')
  for k, fl in enumerate(_fuse_lines(svf)):
      lm = np.zeros((svf.kernel_size, svf.kernel_size), dtype='bool')
      lm[fl[:, 0], fl[:, 1]] = True
      lines_mask[k] = lm
  return lines_mask
  # print(lines_mask[8].astype('int'))

def calculate(svf:SVF):

  # Levantando os parâmetros para o calculo
  pixel_max_size = math.ceil(svf.max_radius / math.floor(svf.kernel_size / 2) * max(svf.mds_src.res))
  downscale_times = math.ceil(math.log2(pixel_max_size / max(svf.mds_src.res)) + 1)
  itens_quantity_by_scale = int(((svf.kernel_size//2) + 1) / 2)
  kernel_border_quantity = (svf.kernel_size * 4) - 4
  pixels_per_angle = downscale_times * itens_quantity_by_scale + itens_quantity_by_scale
  fuse_lines = _fuse_lines(svf)
  # print(itens_quantity_by_scale)

  # Abre o MDS, com PADs
  mds =  svf.mds_src.read(1).astype('float16')
  mdss, transformations = [], []

  # Arranjo da array de mascaras
  masks = lines_mask(svf)
  a_masks = np.argwhere(masks).reshape(kernel_border_quantity,itens_quantity_by_scale * 2,3)[:,:,1:]

  # Cria um arquivo em disco para persistir os calculos
  result_temp = zarr.open(
    '../tmp/result_temp.zarr', 
    mode='a', 
    shape=(svf.mds_src.width, 
          svf.mds_src.height, 
          kernel_border_quantity, 
          pixels_per_angle), 
    chunks=(50,50), 
    dtype=np.float16
  )

  distance_rows = np.zeros((kernel_border_quantity, pixels_per_angle), dtype='float16')
  result_temp = zarr.zeros((svf.mds_src.width, svf.mds_src.height,kernel_border_quantity,pixels_per_angle))

  for i in np.arange(downscale_times):

    resolution = tuple([r * (2 ** i) for r in svf.mds_src.res])

    # Calcula arquivo
    mds, transform = warp.reproject(source=svf.mds_src.read(1),
                                    src_transform=svf.mds_src.transform,
                                    src_crs=svf.mds_src.crs,
                                    dst_crs=svf.mds_src.crs,
                                    dst_nodata=svf.mds_src.nodata,
                                    dst_resolution=resolution,
                                    resampling=Resampling.max)

    # Adiciona PAD
    mds, transform = rasterio.pad(np.squeeze(mds), transform, svf.kernel_size // 2, mode='reflect')
    mds = mds.astype('float16')

    # Forma janelas do tamanho dos Kernels definidos
    windows = view_as_windows(mds, (svf.kernel_size, svf.kernel_size))
    points = np.tile(windows[:,:, svf.kernel_size//2, svf.kernel_size//2].reshape(mds.shape[0] - svf.kernel_size + 1, mds.shape[1] - svf.kernel_size + 1,1,1), (1,1,kernel_border_quantity,1))
    
    # Adicionar as linhas no arquivo de resultados temporarios        
    if i == 0:
      rows = windows[:, :, a_masks[:, :, 0], a_masks[:, :, 1]]
      index = (0, itens_quantity_by_scale * 2)
      print(index)
      # Array de distancias em linhas
      distance_rows[:, index[0]:index[1]] = distance_matrix(size=svf.kernel_size)[fuse_lines[:, :, 0], fuse_lines[:, :, 1]][:, :] * resolution[0]
      # result_temp[:, :, :, index[0]:index[1]] = rows
    else:
      index = (int(i * itens_quantity_by_scale) + itens_quantity_by_scale, int(i * itens_quantity_by_scale) + itens_quantity_by_scale * 2)
      print(index)
      # Array de distancias em linhas
      distance_rows[:, index[0]:index[1]] = distance_matrix(size=svf.kernel_size)[fuse_lines[:, :, 0], fuse_lines[:, :, 1]][:, 3:6] * resolution[0]
      windows = np.tile(windows, (2**i, 2**i, 1, 1))
      points = np.tile(points, (2**i, 2**i, 1, 1))[0:svf.mds_src.width, 0:svf.mds_src.width, :, :]
      # break
      rows = windows[:, :, a_masks[:, :, 0], a_masks[:, :, 1]][0:svf.mds_src.width, 0:svf.mds_src.width, :, itens_quantity_by_scale:]

    # Adiciona o calculo do angulo aos resultados temporários
    result_temp[:, :, :, index[0]:index[1]] = np.arctan2(rows - points, distance_rows[:, index[0]:index[1]])

    ## TODO
    # Achar o angulo máximo ou ArgMax
    # np.max(result_temp[:500, :500, :, 1:], axis=3)

    # Persistir o dado



  return result_temp



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