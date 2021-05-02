'''
inner_surface
Compute inner surfaces for a mesh by throwing streamlines from each
vertex down a signed distance function. Can return inner surfaces
at several steps.
'''

import sys
import numpy as np
import igl
import streamlines3d as s3

def truncated_endpoints(result, depth):
  '''truncate streamlines to a fixed length'''
  endpoints = []
  for re in result:
    streamline = re[::-1]
    dlength = np.sum((streamline[1:]-streamline[:-1])**2, axis=1)**0.5
    length = np.cumsum(dlength)

    sup_array = np.where(length >= depth)[0]
    sup_index = sup_array[0] if len(sup_array)>0 else len(length)-1

    sup_length = length[sup_index]
    inf_length = length[sup_index-1] if sup_index > 0 else 0

    sup_point = streamline[sup_index+1]
    inf_point = streamline[sup_index]

    t = (depth-inf_length)/(sup_length - inf_length)

    point = (1-t)*inf_point + t*sup_point

    endpoints.append(point)

  return np.array(endpoints)

def inner_surface_mesh(v, f, depth=1, nsteps=1, grid_size=0.125, offset=1, gradient_spacing=0.02):
  '''compute an inner surface of mesh by sending streamlines down a signed distance function'''
  # compute signed distance function
  mn = [np.floor(dim-offset) for dim in np.min(v, axis=0)]
  mx = [np.ceil(dim+offset) for dim in np.max(v, axis=0)]
  X, Y, Z = (np.arange(mn[0], mx[0]+grid_size, grid_size),
             np.arange(mn[1], mx[1]+grid_size, grid_size),
             np.arange(mn[2],mx[2]+grid_size, grid_size))
  y, z, x = np.meshgrid(Y, Z, X)
  q = np.column_stack([np.ravel(x), np.ravel(y), np.ravel(z)])
  S, _, _ = igl.signed_distance(q, v, f)
  img = np.zeros((len(X), len(Y), len(Z)))
  ind = ((q-mn)*(1/grid_size)).astype(np.int32)
  img[ind[:, 0], ind[:, 1], ind[:, 2]] = S
  img2 = img[1:-1, 1:-1, 1:-1]

  # compute gradients
  DxW, DyW, DzW = np.gradient(img2, gradient_spacing)

  # compute streamlines
  result = s3.streamlines(
    X[1:-1], Y[1:-1], Z[1:-1],
    DxW, DyW, DzW,
    maxlength=1,
    start_points=v,
    integration_direction="backward")

  # get streamline endpoints after truncating
  endpoints = []
  for step in range(nsteps):
    step_depth = depth * (step + 1)/nsteps
    endpoints_at_depth = truncated_endpoints(result, step_depth)
    endpoints.append(endpoints_at_depth)

  return endpoints

def inner_surface(in_path, out_path, depth=1, nsteps=1):
  '''compute inner surfaces from mesh at in_path, save results'''
  # load input mesh
  v, f = igl.read_triangle_mesh(in_path)

  v2array = inner_surface_mesh(v, f, depth=depth, nsteps=nsteps)

  if nsteps == 1:
    igl.write_triangle_mesh(out_path, v2array[0], f, force_ascii=False)
  else:
    split = out_path.split(".")
    for step in range(nsteps):
      step_str = "%03i"%(step+1)
      out_path_step = ".".join([*(split[:-2]), step_str, split[-1]])
      igl.write_triangle_mesh(out_path_step, v2array[step], f, force_ascii=False)

def main(argv):
  '''convenience inner_surface call from the command line'''
  _, in_path, depth_str, nsteps_str, out_path = argv
  # in_path = "data/derived/trajectories/02-25-07-10-12-19-01/10.ply"
  # depth_str = "1.0"
  # nsteps_str = "1"
  # out_path = "inner-1.ply"

  inner_surface(in_path, out_path, depth=float(depth_str), nsteps=int(nsteps_str))

if __name__ == "__main__":
  main(sys.argv)
