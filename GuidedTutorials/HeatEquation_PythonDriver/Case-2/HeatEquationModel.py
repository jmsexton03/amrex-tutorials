#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import amrex.space3d as amr
import numpy as np


def load_cupy():
    """Load GPU backend if available."""
    if amr.Config.have_gpu:
        try:
            import cupy as cp
            xp = cp
            amr.Print("Note: found and will use cupy")
        except ImportError:
            amr.Print("Warning: GPU found but cupy not available! Trying managed memory in numpy...")
            import numpy as np
            xp = np
        if amr.Config.gpu_backend == "SYCL":
            amr.Print("Warning: SYCL GPU backend not yet implemented for Python")
            import numpy as np
            xp = np

    else:
        import numpy as np
        xp = np
        amr.Print("Note: found and will use numpy")
    return xp


def heat_equation_run(diffusion_coeff=1.0, init_amplitude=1.0, init_width=0.01,
                      n_cell=32, max_grid_size=16, nsteps=100, plot_int=100, dt=1e-5):
    """
    Run heat equation with given parameters and return final state metrics.

    Returns: [max_value, mean_value, std_dev, total_heat, center_value]
    """
    plot_files_output = False
    # CPU/GPU logic
    xp = load_cupy()

    # AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    dom_lo = amr.IntVect(*amr.d_decl(       0,        0,        0))
    dom_hi = amr.IntVect(*amr.d_decl(n_cell-1, n_cell-1, n_cell-1))

    # Make a single box that is the entire domain
    domain = amr.Box(dom_lo, dom_hi)

    # Make BoxArray and Geometry:
    # ba contains a list of boxes that cover the domain,
    # geom contains information such as the physical domain size,
    # number of points in the domain, and periodicity

    # Initialize the boxarray "ba" from the single box "domain"
    ba = amr.BoxArray(domain)
    # Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.max_size(max_grid_size)

    # This defines the physical box, [0,1] in each direction.
    real_box = amr.RealBox([*amr.d_decl( 0., 0., 0.)], [*amr.d_decl( 1., 1., 1.)])

    # This defines a Geometry object
    # periodic in all direction
    coord = 0 # Cartesian
    is_per = [*amr.d_decl(1,1,1)] # periodicity
    geom = amr.Geometry(domain, real_box, coord, is_per);

    # Extract dx from the geometry object
    dx = geom.data().CellSize()

    # Nghost = number of ghost cells for each array
    Nghost = 1

    # Ncomp = number of components for each array
    Ncomp = 1

    # How Boxes are distrubuted among MPI processes
    dm = amr.DistributionMapping(ba)

    # Allocate two phi multifabs: one will store the old state, the other the new.
    phi_old = amr.MultiFab(ba, dm, Ncomp, Nghost)
    phi_new = amr.MultiFab(ba, dm, Ncomp, Nghost)
    phi_old.set_val(0.)
    phi_new.set_val(0.)

    # time = starting time in the simulation
    time = 0.

    # Ghost cells
    ng = phi_old.n_grow_vect
    ngx = ng[0]
    ngy = ng[1]
    ngz = ng[2]

    # Initialize with parameterized initial condition
    for mfi in phi_old:
        bx = mfi.validbox()
        # phiOld is indexed in reversed order (z,y,x) and indices are local
        phiOld = xp.array(phi_old.array(mfi), copy=False)

        x = (xp.arange(bx.small_end[0], bx.big_end[0]+1, 1) + 0.5) * dx[0]
        y = (xp.arange(bx.small_end[1], bx.big_end[1]+1, 1) + 0.5) * dx[1]
        z = (xp.arange(bx.small_end[2], bx.big_end[2]+1, 1) + 0.5) * dx[2]

        rsquared = ((z[:, xp.newaxis, xp.newaxis] - 0.5)**2
                  + (y[xp.newaxis, :, xp.newaxis] - 0.5)**2
                  + (x[xp.newaxis, xp.newaxis, :] - 0.5)**2) / init_width
        phiOld[:, ngz:-ngz, ngy:-ngy, ngx:-ngx] = 1. + init_amplitude * xp.exp(-rsquared)

    # Write a plotfile of the initial data if plot_int > 0
    if plot_int > 0 and plot_files_output:
        step = 0
        pltfile = amr.concatenate("plt", step, 5)
        varnames = amr.Vector_string(['phi'])
        amr.write_single_level_plotfile(pltfile, phi_old, varnames, geom, time, 0)

    # Time evolution
    for step in range(1, nsteps+1):
        phi_old.fill_boundary(geom.periodicity())

        # new_phi = old_phi + dt * Laplacian(old_phi)
        # Loop over boxes
        for mfi in phi_old:
            phiOld = xp.array(phi_old.array(mfi), copy=False)
            phiNew = xp.array(phi_new.array(mfi), copy=False)
            hix = phiOld.shape[3]
            hiy = phiOld.shape[2]
            hiz = phiOld.shape[1]

            # Heat equation with parameterized diffusion
            # Advance the data by dt
            phiNew[:, ngz:-ngz,ngy:-ngy,ngx:-ngx] = (
                phiOld[:, ngz:-ngz,ngy:-ngy,ngx:-ngx] + dt * diffusion_coeff *
                     ((   phiOld[:, ngz  :-ngz     , ngy  :-ngy     , ngx+1:hix-ngx+1]
                       -2*phiOld[:, ngz  :-ngz     , ngy  :-ngy     , ngx  :-ngx     ]
                         +phiOld[:, ngz  :-ngz     , ngy  :-ngy     , ngx-1:hix-ngx-1]) / dx[0]**2
                     +(   phiOld[:, ngz  :-ngz     , ngy+1:hiy-ngy+1, ngx  :-ngx     ]
                       -2*phiOld[:, ngz  :-ngz     , ngy  :-ngy     , ngx  :-ngx     ]
                         +phiOld[:, ngz  :-ngz     , ngy-1:hiy-ngy-1, ngx  :-ngx     ]) / dx[1]**2
                     +(   phiOld[:, ngz+1:hiz-ngz+1, ngy  :-ngy     , ngx  :-ngx     ]
                       -2*phiOld[:, ngz  :-ngz     , ngy  :-ngy     , ngx  :-ngx     ]
                         +phiOld[:, ngz-1:hiz-ngz-1, ngy  :-ngy     , ngx  :-ngx     ]) / dx[2]**2))

        # Update time
        time = time + dt

        # Copy new solution into old solution
        amr.copy_mfab(dst=phi_old, src=phi_new, srccomp=0, dstcomp=0, numcomp=1, nghost=0)
        # Tell the I/O Processor to write out which step we're doing
        # amr.Print(f'Advanced step {step}\n')

        # Write a plotfile of the current data (plot_int was defined in the inputs file)
        if plot_int > 0 and step%plot_int == 0 and plot_files_output:
            pltfile = amr.concatenate("plt", step, 5)
            varnames = amr.Vector_string(['phi'])
            amr.write_single_level_plotfile(pltfile, phi_new, varnames, geom, time, step)

    # Compute output metrics from final state

    # Find center value at (0.5, 0.5, 0.5)
    center_x, center_y, center_z = 0.5, 0.5, 0.5

    # Convert physical coordinates to global cell indices
    i_center = int(center_x / dx[0] - 0.5)
    j_center = int(center_y / dx[1] - 0.5)
    k_center = int(center_z / dx[2] - 0.5)

    center_val = None
    for mfi in phi_new:
        bx = mfi.validbox()

        # Create IntVect3D for the center point
        center_iv = amr.IntVect3D(i_center, j_center, k_center)

        # Check if this box contains the center point
        if bx.contains(center_iv):
            phi_arr = xp.array(phi_new.array(mfi), copy=False)

            # Convert global indices to local array indices
            local_i = i_center - bx.small_end[0] + ngx
            local_j = j_center - bx.small_end[1] + ngy
            local_k = k_center - bx.small_end[2] + ngz

            # Extract center value (array indexed as [z,y,x])
            center_val = float(phi_arr[0, local_k, local_j, local_i])
            if xp.__name__ == 'cupy':
                center_val = float(center_val)
                break

        if center_val is None:
            center_val = 0.0

    # Compute output metrics from final state using PyAMReX built-ins
    max_val = phi_new.max(comp=0, local=False)
    sum_val = phi_new.sum(comp=0, local=False)

    # Get total number of valid cells (excluding ghost zones)
    total_cells = phi_new.box_array().numPts
    mean_val = sum_val / total_cells

    # Use L2 norm for standard deviation calculation
    l2_norm = phi_new.norm2(0)
    sum_sq = l2_norm**2
    variance = (sum_sq / total_cells) - mean_val**2
    std_val = np.sqrt(max(0, variance))

    integral = sum_val * dx[0] * dx[1] * dx[2]

    return np.array([
        max_val,
        mean_val,
        std_val,
        integral,
        center_val
    ])


def parse_inputs():
    """Parse inputs using AMReX ParmParse interface."""
    pp = amr.ParmParse("")

    # Add inputs file if it exists
    import os
    inputs_file = "inputs"
    if os.path.exists(inputs_file):
        pp.addfile(inputs_file)

    # Read simulation parameters with defaults
    n_cell = 32
    pp.query("n_cell", n_cell)

    max_grid_size = 16
    pp.query("max_grid_size", max_grid_size)

    nsteps = 1000
    pp.query("nsteps", nsteps)

    plot_int = 100
    pp.query("plot_int", plot_int)

    dt = 1.0e-5
    pp.query("dt", dt)

    # Read heat equation model parameters with defaults
    diffusion_coeff = 1.0
    pp.query("diffusion_coeff", diffusion_coeff)

    init_amplitude = 1.0
    pp.query("init_amplitude", init_amplitude)

    init_width = 0.01
    pp.query("init_width", init_width)

    return {
        'n_cell': n_cell,
        'max_grid_size': max_grid_size,
        'nsteps': nsteps,
        'plot_int': plot_int,
        'dt': dt,
        'diffusion_coeff': diffusion_coeff,
        'init_amplitude': init_amplitude,
        'init_width': init_width
    }

class HeatEquationModel:
    """Simple wrapper to make heat equation callable with parameter arrays."""

    def __init__(self, n_cell=32, max_grid_size=16, nsteps=1000, plot_int=100, dt=1e-5, use_parmparse=False):
        if use_parmparse:
            # Conditionally initialize AMReX first if using ParmParse
            if not amr.initialized():
                amr.initialize([])

            # Parse inputs from file
            params = parse_inputs()
            self.n_cell = params['n_cell']
            self.max_grid_size = params['max_grid_size']
            self.nsteps = params['nsteps']
            self.plot_int = params['plot_int']
            self.dt = params['dt']
        else:
            self.n_cell = n_cell
            self.max_grid_size = max_grid_size
            self.nsteps = nsteps
            self.plot_int = plot_int
            self.dt = dt

            # Conditionally initialize AMReX
            if not amr.initialized():
                amr.initialize([])

    def __call__(self, params):
        """
        Run heat equation for each parameter set.

        Parameters:
        -----------
        params : numpy.ndarray of shape (n_samples, 3)
            params[:, 0] = diffusion coefficient
            params[:, 1] = initial condition amplitude
            params[:, 2] = initial condition width
            (Use get_pnames() to get these names programmatically)

        Returns:
        --------
        numpy.ndarray of shape (n_samples, 5)
            [max, mean, std, integral, center] for each sample
            (Use get_outnames() to get these names programmatically)
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)

        n_samples = params.shape[0]
        outputs = np.zeros((n_samples, 5))

        for i in range(n_samples):
            outputs[i, :] = heat_equation_run(
                diffusion_coeff=params[i, 0],
                init_amplitude=params[i, 1],
                init_width=params[i, 2],
                n_cell=self.n_cell,
                max_grid_size=self.max_grid_size,
                nsteps=self.nsteps,
                plot_int=self.plot_int,
                dt=self.dt
            )

        return outputs

    def get_pnames(self):
        """
        Get parameter names for the heat equation model.

        Returns:
        --------
        list : Parameter names corresponding to the input dimensions
        """
        return ["diffusion coefficient", "initial condition amplitude", "initial condition width"]

    def get_outnames(self):
        """
        Get output names for the heat equation model.

        Returns:
        --------
        list : Output names corresponding to the computed quantities
        """
        return ["max", "mean", "std", "integral", "center"]

if __name__ == "__main__":

    # Initialize AMReX
    amr.initialize([])

    # Create model using ParmParse to read from inputs file
    model = HeatEquationModel(use_parmparse=True)

    print(f"Heat equation model initialized with:")
    print(f"  n_cell = {model.n_cell}")
    print(f"  max_grid_size = {model.max_grid_size}")
    print(f"  nsteps = {model.nsteps}")
    print(f"  plot_int = {model.plot_int}")
    print(f"  dt = {model.dt}")

    # Test with random parameters
    test_params = np.array([
        [1.0, 1.0, 0.01],   # baseline
        [2.0, 1.5, 0.02],   # higher diffusion, higher amplitude
        [0.5, 2.0, 0.005]   # lower diffusion, higher amplitude, narrower
    ])

    print("\nRunning heat equation with parameters:")
    print("  [diffusion, amplitude, width]")
    print(test_params)

    outputs = model(test_params)

    print("\nResults [max, mean, std, integral, center]:")
    print(outputs)

    # Finalize AMReX
    amr.finalize()

