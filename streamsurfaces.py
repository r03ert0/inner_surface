import numpy as np
import igl
from scipy.sparse import diags
import scipy.sparse.linalg as spla

def streamsurfaces(
    verts, tris,
    x, y, z, u, v, w,
    num_steps=100,
    smoothness=0.1 ):
    '''
    Compute the stream surface of a vector field. Integration direction
    is always forward.

        Parameters
    ----------
    verts: array of float with shape (, 3)
        Mesh vertices
    u, v, w: array of float with shape (, )
        Vector field components
    tris: array of int with shape (, 3)
        Mesh triangles
    num_steps: int
        Number of time steps to take to compute the stream surface
    Returns
    -------
    verts: array of float with shape (, 3)
        Stream surface vertices
    '''

    grid = Grid(x, y, z)
    dmap = DomainMap(grid)

    integrate_step = get_integrator(u, v, w, dmap)

    sp3 = np.asanyarray(verts, dtype=float).copy()

    # Convert start_points from data to array coords
    # Shift the seed points from the bottom left of the data so that
    # data2grid works properly.
    sp3[:, 0] -= grid.x_origin
    sp3[:, 1] -= grid.y_origin
    sp3[:, 2] -= grid.z_origin

    # Compute cotangent weights for the mesh laplacian
    L = igl.cotmatrix(sp3, tris)
    M = igl.massmatrix(sp3, tris, igl.MASSMATRIX_TYPE_VORONOI)
    A = M - smoothness * L

    # Integrate the surface over time
    sp3_array = [sp3 + [grid.x_origin, grid.y_origin, grid.z_origin]]
    for _ in range(num_steps):

        xg, yg, zg = dmap.data2grid(sp3[:, 0], sp3[:, 1], sp3[:, 2])
        disp = integrate_step(xg, yg, zg)

        # smooth implicit
        disp_smooth = np.zeros_like(disp)
        for j in range(3):
            rhs = M @ disp[:, j]
            disp_smooth[:, j] = spla.spsolve(A, rhs)

        # advect the vertices
        sp3 += disp_smooth

        sp3_array.append(sp3 + [grid.x_origin, grid.y_origin, grid.z_origin])

    return sp3_array


# Coordinate definitions
# ========================

class DomainMap:
    """
    Map representing different coordinate systems.

    Coordinate definitions:

    * axes-coordinates goes from 0 to 1 in the domain.
    * data-coordinates are specified by the input x-y coordinates.
    * grid-coordinates goes from 0 to N and 0 to M for an N x M grid,
      where N and M match the shape of the input data.
    * mask-coordinates goes from 0 to N and 0 to M for an N x M mask,
      where N and M are user-specified to control the density of streamlines.

    This class also has methods for adding trajectories to the StreamMask.
    Before adding a trajectory, run `start_trajectory` to keep track of regions
    crossed by a given trajectory. Later, if you decide the trajectory is bad
    (e.g., if the trajectory is very short) just call `undo_trajectory`.
    """

    def __init__(self, grid):
        self.grid = grid

        self.x_data2grid = 1. / grid.dx
        self.y_data2grid = 1. / grid.dy
        self.z_data2grid = 1. / grid.dz

    def data2grid(self, xd, yd, zd):
        return xd * self.x_data2grid, yd * self.y_data2grid, zd * self.z_data2grid

    def grid2data(self, xg, yg, zg):
        return xg / self.x_data2grid, yg / self.y_data2grid, zg / self.z_data2grid

    def update_trajectory(self, xg, yg, zg):
        if not self.grid.within_grid(xg, yg, zg):
            raise InvalidIndexError

class Grid:
    """Grid of data."""
    def __init__(self, x, y, z):

        self.nx = len(x)
        self.ny = len(y)
        self.nz = len(z)

        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dz = z[1] - z[0]

        self.x_origin = x[0]
        self.y_origin = y[0]
        self.z_origin = z[0]

        self.width = x[-1] - x[0]
        self.height = y[-1] - y[0]
        self.depth = z[-1] - z[0]

    @property
    def shape(self):
        return self.nx, self.ny, self.nz

    def within_grid(self, xi, yi, zi):
        """Return whether (*xi*, *yi*, *zi*) is a valid index of the grid."""
        # Note that xi/yi/zi can be floats; so, for example, we can't simply check
        # `xi < self.nx` since *xi* can be `self.nx - 1 < xi < self.nx`

        return min(xi)>=0 and max(xi) <= self.nx - 1 and min(yi) >= 0 and max(yi) <= self.ny - 1 and min(zi) >= 0 and max(zi) <= self.nz - 1

class InvalidIndexError(Exception):
    pass


class TerminateTrajectory(Exception):
    pass

# Integrator definitions
# =======================

def get_integrator(u, v, w, dmap) # minlength, maxlength, integration_direction

    # rescale velocity onto grid-coordinates for integrations.
    u, v, w = dmap.data2grid(u, v, w)

    # speed (path length) will be in axes-coordinates
    u_ax = u / (dmap.grid.nx - 1)
    v_ax = v / (dmap.grid.ny - 1)
    w_ax = w / (dmap.grid.nz - 1)
    speed = np.ma.sqrt(u_ax ** 2 + v_ax ** 2 + w_ax ** 2)

    def forward_time(xi, yi, zi):
        if not dmap.grid.within_grid(xi, yi, zi):
            raise OutOfBounds
        ds_dt = interpgrid(speed, xi, yi, zi)
        # if min(ds_dt) == 0:
        #     raise TerminateTrajectory()
        dt_ds = 1. / ds_dt
        ui = interpgrid(u, xi, yi, zi)
        vi = interpgrid(v, xi, yi, zi)
        wi = interpgrid(w, xi, yi, zi)

        return ui * dt_ds, vi * dt_ds, wi * dt_ds

    def integrate_step(x0, y0, z0):
        """
        Return x, y, z grid-coordinates of trajectory based on starting point.

        Integrate both forward and backward in time from starting point in
        grid coordinates.

        Integration is terminated when a trajectory reaches a domain boundary.
        The resulting trajectory is None if it is shorter than `minlength`.
        """

        disp = _integrate_rk12_step(x0, y0, z0, dmap, forward_time)
        return disp

    return integrate_step


class OutOfBounds(IndexError):
    pass


def _integrate_rk12_step(x0, y0, z0, dmap, f):
    """
    2nd-order Runge-Kutta algorithm with adaptive step size.

    This method is also referred to as the improved Euler's method, or Heun's
    method. This method is favored over higher-order methods because:

    1. To get decent looking trajectories and to sample every mask cell
       on the trajectory we need a small timestep, so a lower order
       solver doesn't hurt us unless the data is *very* high resolution.
       In fact, for cases where the user inputs
       data smaller or of similar grid size to the mask grid, the higher
       order corrections are negligible because of the very fast linear
       interpolation used in `interpgrid`.

    2. For high resolution input data (i.e. beyond the mask
       resolution), we must reduce the timestep. Therefore, an adaptive
       timestep is more suited to the problem as this would be very hard
       to judge automatically otherwise.

    This integrator is about 1.5 - 2x as fast as both the RK4 and RK45
    solvers in most setups on my machine. I would recommend removing the
    other two to keep things simple.

    Parameters
    ----------
    x0, y0, z0 : array of floats
        Initial position.
    dmap : `~sunpy.map.GenericMap`
        The map on which to perform the integration.
    f : function
        Function to compute the gradient at a given point.

    Results
    -------
    disp: array of floats
        The displacement of the trajectory in grid coordinates.
    """
    # This error is below that needed to match the RK4 integrator. It
    # is set for visual reasons -- too low and corners start
    # appearing ugly and jagged. Can be tuned.
    maxerror = 0.003
    maxds = 0.001

    ds = maxds
    stotal = 0

    # the step
    # Compute the two intermediate gradients.
    # f should raise OutOfBounds if the locations given are
    # outside the grid.
    k1x, k1y, k1z = f(x0, y0, z0)
    k2x, k2y, k2z = f(x0 + ds * k1x, y0 + ds * k1y, z0 + ds * k1z)

    dx2 = ds * 0.5 * (k1x + k2x)
    dy2 = ds * 0.5 * (k1y + k2y)
    dz2 = ds * 0.5 * (k1z + k2z)

    disp = np.column_stack((dx2, dy2, dz2))

    return disp

# def _euler_step(xf_traj, yf_traj, zf_traj, dmap, f):
#     """Simple Euler integration step that extends streamline to boundary."""
#     nz, ny, nx = dmap.grid.shape
#     xi = xf_traj[-1]
#     yi = yf_traj[-1]
#     zi = zf_traj[-1]
#     cx, cy, cz = f(xi, yi, zi)

#     if cx == 0:
#         dsx = np.inf
#     elif cx < 0:
#         dsx = xi / -cx
#     else:
#         dsx = (nx - 1 - xi) / cx

#     if cy == 0:
#         dsy = np.inf
#     elif cy < 0:
#         dsy = yi / -cy
#     else:
#         dsy = (ny - 1 - yi) / cy

#     if cz == 0:
#         dsz = np.inf
#     elif cz < 0:
#         dsz = zi / -cz
#     else:
#         dsz = (nz - 1 - zi) / cz

#     ds = min(dsx, dsy, dsz)
#     xf_traj.append(xi + cx * ds)
#     yf_traj.append(yi + cy * ds)
#     zf_traj.append(zi + cz * ds)

#     return ds, xf_traj, yf_traj, zf_traj

# Utility functions
# ========================

def interpgrid(a, xi, yi, zi):
    """Fast 3D, linear interpolation on an integer grid"""

    # Nz, Ny, Nx = np.shape(a)
    Nx, Ny, Nz = np.shape(a)
    x = xi.astype(int)
    y = yi.astype(int)
    z = zi.astype(int)

    # Check that xn, yn, zn don't exceed max index
    xn = np.clip(x + 1, 0, Nx - 1)
    yn = np.clip(y + 1, 0, Ny - 1)
    zn = np.clip(z + 1, 0, Nz - 1)

    a000 = a[x, y, z]
    a100 = a[xn, y, z]
    a010 = a[x, yn, z]
    a110 = a[xn, yn, z]
    a001 = a[x, y, zn]
    a101 = a[xn, y, zn]
    a011 = a[x, yn, zn]
    a111 = a[xn, yn, zn]

    xt = xi - x
    yt = yi - y
    zt = zi - z

    return (a000 * (1 - xt)*(1 - yt)*(1 - zt)
          + a100 * xt * (1 - yt) * (1 - zt) +
          + a010 * (1 - xt) * yt * (1 - zt) +
          + a001 * (1 - xt) * (1 - yt) * zt +
          + a101 * xt * (1 - yt) * zt +
          + a011 * (1 - xt) * yt * zt +
          + a110 * xt * yt * (1 - zt) +
          + a111 * xt * yt * zt)
