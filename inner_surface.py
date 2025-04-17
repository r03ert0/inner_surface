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
import streamsurfaces as ss
import nibabel as nib

def truncated_streamlines(raw_streamlines, depth):
  '''truncate streamlines to a fixed length'''
  endpoints = []
  for rs in raw_streamlines:
    streamline = rs[::-1]
    dlength = np.sum((streamline[1:]-streamline[:-1])**2, axis=1)**0.5
    length = np.cumsum(dlength)

    sup_array = np.where(length >= depth)[0]
    sup_index = sup_array[0] if len(sup_array)>0 else len(length)-1

    sup_length = length[sup_index] if sup_index>=0 else 0
    inf_length = length[sup_index-1] if sup_index > 0 else 0

    sup_point = streamline[sup_index+1]
    inf_point = streamline[sup_index]

    if sup_length == inf_length:
      t = 1
    else:
      t = (depth-inf_length)/(sup_length - inf_length)

    point = (1-t)*inf_point + t*sup_point

    endpoints.append(point)

  return np.array(endpoints)

def compute_sdf_grid(v, offset=1, grid_size=0.02):
  '''compute a regular grid, used for the SDF'''
  mn = [np.floor(dim-offset) for dim in np.min(v, axis=0)]
  mx = [np.ceil(dim+offset) for dim in np.max(v, axis=0)]
  X, Y, Z = (np.arange(mn[0], mx[0]+grid_size, grid_size),
             np.arange(mn[1], mx[1]+grid_size, grid_size),
             np.arange(mn[2], mx[2]+grid_size, grid_size))
  return X, Y, Z, mn

def compute_sdf(v, f, X, Y, Z, minimum, grid_size=0.02):
  '''compute signed distance function'''
  y, z, x = np.meshgrid(Y, Z, X)
  q = np.column_stack([np.ravel(x), np.ravel(y), np.ravel(z)])
  S, _, _ = igl.signed_distance(q, v, f)
  img = np.zeros((len(X), len(Y), len(Z)))
  ind = ((q-minimum)*(1/grid_size)).astype(np.int32)
  img[ind[:, 0], ind[:, 1], ind[:, 2]] = S
  # img2 = img[offset:-offset, offset:-offset, offset:-offset]
  # return img2
  return img

def compute_streamlines(v, X, Y, Z, vol, gradient_spacing=0.02, integration_direction="backward"):
  '''Compute streamlines from mesh vertices
  Parameters
  ----------
  v: array of float with shape (, 3)
    Mesh vertices
  X, Y, Z: array of float with shape (W, ) and (H, ) and (D, )
    Grid used for the computation of the signed distance function.
  vol: array of float with shape (W, H, D)
    Signed distance function.
  gradient_spacing: float
    Size of the steps used for integration of the SDF gradient.
  integration_direction: str
    Direction of integration of the SDF gradient. Can be "forward", "backward" or "both".
  '''
  # compute gradients
  DxW, DyW, DzW = np.gradient(vol, gradient_spacing)

  # compute streamlines
  raw_streamlines = s3.streamlines(
    X, Y, Z, # X[1:-1], Y[1:-1], Z[1:-1],
    DxW, DyW, DzW,
    minlength=0,
    maxlength=1,
    start_points=v,
    integration_direction=integration_direction)

  return raw_streamlines

def compute_streamsurfaces(v, f, X, Y, Z, field, gradient_spacing=0.02, num_steps=20, integration_direction="backward"):
  '''Compute streamsurfaces from mesh vertices'''
  # compute gradients
  DxW, DyW, DzW = np.gradient(field, gradient_spacing)
  # nib.save(nib.Nifti1Image(DxW, np.eye(4)), "9c_test_DxW.nii.gz")
  # nib.save(nib.Nifti1Image(DyW, np.eye(4)), "9c_test_DyW.nii.gz")
  # nib.save(nib.Nifti1Image(DzW, np.eye(4)), "9c_test_DzW.nii.gz")

  # compute streamsurfaces
  raw_streamsurfaces = ss.streamsurfaces(
    v, f,
    X, Y, Z,
    DxW, DyW, DzW,
    minlength=0,
    maxlength=1,
    smoothness=0.3,
    num_steps=num_steps,
    integration_direction=integration_direction)
  
  return raw_streamsurfaces

def conform_streamlines(raw_streamlines, depth, nsteps):
  '''get streamlines with fixed-length steps'''
  streamlines = []
  for step in range(nsteps):
    step_depth = depth * (step + 1)/nsteps
    streamlines_at_depth = truncated_streamlines(raw_streamlines, step_depth)
    streamlines.append(streamlines_at_depth)

  return streamlines

def load_sdf(sdf_path):
  '''Load a nifti volume containing SDF data'''
  sdf = nib.load(sdf_path).get_fdata()

  return sdf

def load_raw_streamlines(raw_streamlines_path):
  '''Load raw streamlines'''
  raw_streamlines = np.load(raw_streamlines_path, allow_pickle=True)["raw_streamlines"]

  return raw_streamlines

def inner_surface_mesh(v, f, depth=1, nsteps=1, grid_size=0.125, offset=1,
  gradient_spacing=0.02, precomputed_sdf=None,
  precomputed_raw_streamlines=None, save_sdf_path=None, save_raw_streamlines_path=None):
  '''Compute an inner surface of mesh by sending streamlines down a signed
  distance function.

  Parameters
  ----------

  v: array of float with shape (, 3)
    Mesh vertices
  f: array of int with shape (, 3)
    Mesh triangles
  depth: float
    Depth of the inner surface
  nsteps: int
    Number of steps to produce from the original surface to the inner surface.
  grid_size: float
    Size of the grid used for the computation of the signed distance function.
  offset: int
    Number of voxels to add around the volume containing the mesh.
  gradient_spacing: float
    Size of the steps used for integration of the SDF gradient.

  Returns
  -------
  streamlines: list of arrays of float with shape (, 3)
    List of `nsteps` inner surface's vertices.
  '''

  X, Y, Z, minimum = compute_sdf_grid(v, offset, grid_size)

  if precomputed_sdf is not None:
    img2 = precomputed_sdf
  else:
    img2 = compute_sdf(v, f, X, Y, Z, minimum, grid_size)

    if save_sdf_path:
      affine = np.eye(4)
      affine[0, 0] = grid_size
      affine[1, 1] = grid_size
      affine[2, 2] = grid_size
      affine[3, 3] = 1
      vol = nib.Nifti1Image(img2, affine=affine)
      nib.save(vol, save_sdf_path)

  if precomputed_raw_streamlines is not None:
    raw_streamlines = precomputed_raw_streamlines
  else:
    raw_streamlines = compute_streamlines(v, X, Y, Z, img2, gradient_spacing)

    if save_raw_streamlines_path:
      np.savez_compressed(
        save_raw_streamlines_path,
        allow_pickle=True,
        raw_streamlines=raw_streamlines)

  streamlines = conform_streamlines(raw_streamlines, depth, nsteps)

  return streamlines

def save_inner_surface_mesh(v_array, f, out_path):
  '''Save inner surface meshes.

  Parameters
  ----------

  v_array: list of arrays of floats with shape (, 3)
    List of mesh vertices for a series of inner meshes
  f: array of int with shape (, 3)
    Mesh triangles
  out_path: string
    Path for saving the series of meshes
  '''

  nsteps = len(v_array)
  if nsteps == 1:
    igl.write_triangle_mesh(out_path, v_array[0], f, force_ascii=False)
  else:
    split = out_path.split(".")
    for step in range(nsteps):
      step_str = "%03i"%(step+1)
      out_path_step = ".".join([*(split[:-2]), step_str, split[-1]])
      igl.write_triangle_mesh(out_path_step, v_array[step], f, force_ascii=False)

def inner_surface(in_path, out_path, depth=1, nsteps=1):
  '''Compute inner surfaces from mesh at in_path, save results.

  Parameters
  ----------

  in_path: string
    Path to the input mesh.
  out_path: string
    Path to the output mesh
  depth: float
    Depth of the inner surface
  nsteps: int
    Number of steps to produce from the original surface to the inner surface.

  Returns
  -------
  Nothing returned.
  '''
  # load input mesh
  v, f = igl.read_triangle_mesh(in_path)

  v_array = inner_surface_mesh(v, f, depth=depth, nsteps=nsteps)

  save_inner_surface_mesh(v_array, f, out_path)

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
