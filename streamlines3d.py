"""
3D streamlines, based on matplotlib's streamplot.py

"""

import numpy as np


def streamlines(x, y, z, u, v, w, minlength=0.1, start_points=None, maxlength=4.0, integration_direction='both'):
    """
    Draw streamlines of a vector flow.

    Parameters
    ----------
    x, y, z : 1D arrays
        Evenly spaced strictly increasing arrays to make a grid.
    u, v, w : 3D arrays
        *x*, *y* and *z*-velocities.
    minlength : float
        Minimum length of streamline in axes coordinates.
    start_points : Nx3 array
        Coordinates of starting points for the streamlines in data coordinates
        (the same coordinates as the *x* and *y* arrays).
    maxlength : float
        Maximum length of streamline in axes coordinates.
    integration_direction : {'forward', 'backward', 'both'}, default: 'both'
        Integrate the streamline in forward, backward or both directions.

    Returns
    -------
    array
    """
    grid = Grid(x, y, z)
    dmap = DomainMap(grid)

    if integration_direction == 'both':
        maxlength /= 2.

    integrate = get_integrator(u, v, w, dmap, minlength, maxlength, integration_direction)

    trajectories = []
    sp3 = np.asanyarray(start_points, dtype=float).copy()

    # Check if start_points are outside the data boundaries
    for xs, ys, zs in sp3:
        if not (grid.x_origin <= xs <= grid.x_origin + grid.width and
                grid.y_origin <= ys <= grid.y_origin + grid.height and
                grid.z_origin <= zs <= grid.z_origin + grid.depth):
            raise ValueError("Starting point ({}, {}, {}) outside of data "
                              "boundaries".format(xs, ys, zs))

    # Convert start_points from data to array coords
    # Shift the seed points from the bottom left of the data so that
    # data2grid works properly.
    sp3[:, 0] -= grid.x_origin
    sp3[:, 1] -= grid.y_origin
    sp3[:, 2] -= grid.z_origin

    for xs, ys, zs in sp3:
        xg, yg, zg = dmap.data2grid(xs, ys, zs)
        t = integrate(xg, yg, zg)
        if t is not None:
            trajectories.append(t)

    streamlines_array = []
    for t in trajectories:
        # Rescale from grid-coordinates to data-coordinates.
        tx, ty, tz = dmap.grid2data(*np.array(t))
        tx += grid.x_origin
        ty += grid.y_origin
        tz += grid.z_origin

        # points = np.transpose([tx, ty, tz]).reshape(-1, 1, 3)
        # streamlines_array.extend(np.hstack([points[:-1], points[1:]]))
        streamlines_array.append(np.column_stack([tx, ty, tz]))
    # print(streamlines_array[0][:10])

    return streamlines_array

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

        # for attr in dir(self):
        #   print("%s = %r" % (attr, getattr(self, attr)))

    @property
    def shape(self):
        return self.nz, self.ny, self.nx

    def within_grid(self, xi, yi, zi):
        """Return whether (*xi*, *yi*, *zi*) is a valid index of the grid."""
        # Note that xi/yi/zi can be floats; so, for example, we can't simply check
        # `xi < self.nx` since *xi* can be `self.nx - 1 < xi < self.nx`
        return 0 <= xi <= self.nx - 1 and 0 <= yi <= self.ny - 1 and 0 <= zi <= self.nz - 1

class InvalidIndexError(Exception):
    pass


class TerminateTrajectory(Exception):
    pass


# Integrator definitions
# =======================

def get_integrator(u, v, w, dmap, minlength, maxlength, integration_direction):

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
        if ds_dt == 0:
            raise TerminateTrajectory()
        dt_ds = 1. / ds_dt
        ui = interpgrid(u, xi, yi, zi)
        vi = interpgrid(v, xi, yi, zi)
        wi = interpgrid(w, xi, yi, zi)
        return ui * dt_ds, vi * dt_ds, wi * dt_ds

    def backward_time(xi, yi, zi):
        dxi, dyi, dzi = forward_time(xi, yi, zi)
        return -dxi, -dyi, -dzi

    def integrate(x0, y0, z0):
        """
        Return x, y, z grid-coordinates of trajectory based on starting point.

        Integrate both forward and backward in time from starting point in
        grid coordinates.

        Integration is terminated when a trajectory reaches a domain boundary.
        The resulting trajectory is None if it is shorter than `minlength`.
        """

        stotal, x_traj, y_traj, z_traj = 0., [], [], []

        if integration_direction in ['both', 'backward']:
            s, xt, yt, zt = _integrate_rk12(x0, y0, z0, dmap, backward_time, maxlength)
            stotal += s
            x_traj += xt[::-1]
            y_traj += yt[::-1]
            z_traj += zt[::-1]

        if integration_direction in ['both', 'forward']:
            s, xt, yt, zt = _integrate_rk12(x0, y0, z0, dmap, forward_time, maxlength)
            if len(x_traj) > 0:
                xt = xt[1:]
                yt = yt[1:]
                zt = zt[1:]
            stotal += s
            x_traj += xt
            y_traj += yt
            z_traj += zt

        if stotal >= minlength:
            return x_traj, y_traj, z_traj
        else:  # reject short trajectories
            return None

    return integrate


class OutOfBounds(IndexError):
    pass


def _integrate_rk12(x0, y0, z0, dmap, f, maxlength):
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
    """
    # This error is below that needed to match the RK4 integrator. It
    # is set for visual reasons -- too low and corners start
    # appearing ugly and jagged. Can be tuned.
    maxerror = 0.003
    maxds = 0.1

    ds = maxds
    stotal = 0
    xi = x0
    yi = y0
    zi = z0
    xf_traj = []
    yf_traj = []
    zf_traj = []

    # stotal2 = 0

    while True:
        # print("stotal2:", stotal2, "stotal:", stotal, "xi,yi,zi:", xi,yi,zi)
        try:
            if not dmap.grid.within_grid(xi, yi, zi):
                raise OutOfBounds

            xf_traj.append(xi)
            yf_traj.append(yi)
            zf_traj.append(zi)
            # Compute the two intermediate gradients.
            # f should raise OutOfBounds if the locations given are
            # outside the grid.
            k1x, k1y, k1z = f(xi, yi, zi)
            k2x, k2y, k2z = f(xi + ds * k1x, yi + ds * k1y, zi + ds * k1z)

        except OutOfBounds:
            # Out of the domain during this step.
            # Take an Euler step to the boundary to improve neatness
            # unless the trajectory is currently empty.
            if xf_traj:
                ds, xf_traj, yf_traj, zf_traj = _euler_step(xf_traj, yf_traj, zf_traj, dmap, f)
                stotal += ds
            break
        except TerminateTrajectory:
            break

        dx1 = ds * k1x
        dy1 = ds * k1y
        dz1 = ds * k1z
        dx2 = ds * 0.5 * (k1x + k2x)
        dy2 = ds * 0.5 * (k1y + k2y)
        dz2 = ds * 0.5 * (k1z + k2z)

        # ds2 = (dx2**2+dy2**2+dz2**2)**0.5

        # nx, ny, nz = dmap.grid.shape
        nz, ny, nx = dmap.grid.shape
      
        # Error is normalized to the axes coordinates
        error = (((dx2 - dx1) / (nx - 1))**2 + ((dy2 - dy1) / (ny - 1))**2 + ((dz2 - dz1) / (nz - 1))**2)**0.5

        # Only save step if within error tolerance
        if error < maxerror:
            xi += dx2
            yi += dy2
            zi += dz2

            try:
                dmap.update_trajectory(xi, yi, zi)
            except InvalidIndexError:
                break

            if stotal + ds > maxlength:
                break

            # if stotal2 + ds2 > maxlength:
            #     break

            stotal += ds

            # stotal2 += ds2

        # recalculate stepsize based on step error
        ds = maxds if error == 0 else min(maxds, 0.85 * ds * (maxerror / error) ** 0.5)

    return stotal, xf_traj, yf_traj, zf_traj
    # return stotal2, xf_traj, yf_traj, zf_traj


def _euler_step(xf_traj, yf_traj, zf_traj, dmap, f):
    """Simple Euler integration step that extends streamline to boundary."""
    nz, ny, nx = dmap.grid.shape
    xi = xf_traj[-1]
    yi = yf_traj[-1]
    zi = zf_traj[-1]
    cx, cy, cz = f(xi, yi, zi)

    if cx == 0:
        dsx = np.inf
    elif cx < 0:
        dsx = xi / -cx
    else:
        dsx = (nx - 1 - xi) / cx

    if cy == 0:
        dsy = np.inf
    elif cy < 0:
        dsy = yi / -cy
    else:
        dsy = (ny - 1 - yi) / cy

    if cz == 0:
        dsz = np.inf
    elif cz < 0:
        dsz = zi / -cz
    else:
        dsz = (nz - 1 - zi) / cz

    ds = min(dsx, dsy, dsz)
    xf_traj.append(xi + cx * ds)
    yf_traj.append(yi + cy * ds)
    zf_traj.append(zi + cz * ds)

    return ds, xf_traj, yf_traj, zf_traj


# Utility functions
# ========================

def interpgrid(a, xi, yi, zi):
    """Fast 3D, linear interpolation on an integer grid"""

    Nz, Ny, Nx = np.shape(a)
    if isinstance(xi, np.ndarray):
        x = xi.astype(int)
        y = yi.astype(int)
        z = zi.astype(int)
        # Check that xn, yn, zn don't exceed max index
        xn = np.clip(x + 1, 0, Nx - 1)
        yn = np.clip(y + 1, 0, Ny - 1)
        zn = np.clip(z + 1, 0, Nz - 1)
    else:
        x = int(xi)
        y = int(yi)
        z = int(zi)
        # conditional is faster than clipping for integers
        xn = x if x == (Nx - 1) else x + 1
        yn = y if y == (Ny - 1) else y + 1
        zn = z if z == (Nz - 1) else z + 1

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

# # test
# def Elevation(x,y,sigma=0.2):
#     return 1 + 8 * np.exp(-(x**2 + y**2)/(2*sigma**2))
# X, Y, Z = np.arange(-1,1,0.02),np.arange(-1,1,0.02),np.arange(-1,1,0.02)
# x, y, z = np.meshgrid(X, Y, Z)
# W = Elevation(x, y)

# DyW, DxW, DzW = np.gradient(W,0.02)

# angle = np.linspace(0, 2*np.pi, 50)
# cx = 0.5*np.cos(angle)
# cy = 0.5*np.sin(angle)
# cz = 0*angle
# start_points = np.column_stack((cx, cy, cz))
# # print(len(start_points))
# result = streamlines(X,Y,Z,DxW,DyW,DzW, maxlength=0.3, start_points=start_points)
# # print(len(result))