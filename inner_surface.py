import sys
import numpy as np
import igl
import streamlines3d as s3

def inner_surface_mesh(v, f, depth=1, step=0.125, offset=1, gradient_spacing=0.02):
    # compute signed distance function
    mn = [np.floor(dim-offset) for dim in np.min(v, axis=0)]
    mx = [np.ceil(dim+offset) for dim in np.max(v, axis=0)]
    X, Y, Z = np.arange(mn[0],mx[0]+step, step), np.arange(mn[1],mx[1]+step, step), np.arange(mn[2],mx[2]+step, step)
    y, z, x = np.meshgrid(Y, Z, X)
    q = np.column_stack([np.ravel(x), np.ravel(y), np.ravel(z)])
    S, I, C = igl.signed_distance(q, v, f)
    img = np.zeros((len(X), len(Y), len(Z)))
    ind = ((q-mn)*(1/step)).astype(np.int32)
    img[ind[:,0],ind[:,1],ind[:,2]] = S
    img2 = img[1:-1,1:-1,1:-1]

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
    c = []
    for re in result:
        er = re[::-1]
        dl = np.sum((er[1:]-er[:-1])**2, axis=1)**0.5
        length = np.cumsum(dl)

        sup_array = np.where(length>=depth)[0]
        sup_index = sup_array[0] if len(sup_array)>0 else len(length)-1

        sup_length = length[sup_index]
        inf_length = length[sup_index-1] if sup_index>0 else 0

        sup_point = er[sup_index+1]
        inf_point = er[sup_index]

        t = (depth-inf_length)/(sup_length - inf_length)

        point = (1-t)*inf_point + t*sup_point

        c.append(point)

    return np.array(c)

def inner_surface(in_path, out_path, depth=1):
    # load input mesh
    v,f = igl.read_triangle_mesh(in_path)
    v2 = inner_surface_mesh(v, f, depth=depth)

    igl.write_triangle_mesh(out_path, v2, f, force_ascii=False)

def main(argv):
    _, in_path, depth_str, out_path = argv
    # in_path = "data/derived/trajectories/02-25-07-10-12-19-01/10.ply"
    # depth_str = "1.0"
    # out_path = "inner-1.ply"

    inner_surface(in_path, out_path, depth=float(depth_str))

if __name__ == "__main__":
    main(sys.argv)
