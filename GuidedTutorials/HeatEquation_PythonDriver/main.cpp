/*
 * Simplified Heat Equation with Python Driver Interface
 *
 * This demonstrates the minimal setup needed for a one-line Python interface
 * that calls C++ simulation code and returns structured results.
 */

#include <AMReX.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>

// Simple result structure for Python interface
struct SimulationResult {
    double max_temperature;
    int final_step;
    double final_time;
    bool success;
};

SimulationResult heat_equation_main(int argc, char* argv[])
{
    SimulationResult result = {0.0, 0, 0.0, false};

    amrex::Initialize(argc,argv);
    {

    // **********************************
    // DECLARE SIMULATION PARAMETERS
    // **********************************

    int n_cell;
    int max_grid_size;
    int nsteps;
    int plot_int;
    amrex::Real dt;

    // **********************************
    // READ PARAMETER VALUES FROM INPUT DATA
    // **********************************
    {
        amrex::ParmParse pp;
        pp.get("n_cell",n_cell);
        pp.get("max_grid_size",max_grid_size);
        nsteps = 10;
        pp.query("nsteps",nsteps);
        plot_int = -1;
        pp.query("plot_int",plot_int);
        pp.get("dt",dt);
    }

    // **********************************
    // DEFINE SIMULATION SETUP AND GEOMETRY
    // **********************************

    amrex::BoxArray ba;
    amrex::Geometry geom;

    amrex::IntVect dom_lo(0,0,0);
    amrex::IntVect dom_hi(n_cell-1, n_cell-1, n_cell-1);

    amrex::Box domain(dom_lo, dom_hi);
    ba.define(domain);
    ba.maxSize(max_grid_size);

    amrex::RealBox real_box({ 0., 0., 0.}, { 1., 1., 1.});
    amrex::Array<int,3> is_periodic{1,1,1};
    geom.define(domain, real_box, amrex::CoordSys::cartesian, is_periodic);

    amrex::GpuArray<amrex::Real,3> dx = geom.CellSizeArray();

    int Nghost = 1;
    int Ncomp = 1;

    amrex::DistributionMapping dm(ba);
    amrex::MultiFab phi_old(ba, dm, Ncomp, Nghost);
    amrex::MultiFab phi_new(ba, dm, Ncomp, Nghost);

    amrex::Real time = 0.0;

    // **********************************
    // INITIALIZE DATA LOOP
    // **********************************

    for (amrex::MFIter mfi(phi_old); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        const amrex::Array4<amrex::Real>& phiOld = phi_old.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::Real x = (i+0.5) * dx[0];
            amrex::Real y = (j+0.5) * dx[1];
            amrex::Real z = (k+0.5) * dx[2];
            amrex::Real rsquared = ((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5))/0.01;
            phiOld(i,j,k) = 1. + std::exp(-rsquared);
        });
    }

    // **********************************
    // WRITE INITIAL PLOT FILE
    // **********************************

    if (plot_int > 0)
    {
        int step = 0;
        const std::string& pltfile = amrex::Concatenate("plt",step,5);
        WriteSingleLevelPlotfile(pltfile, phi_old, {"phi"}, geom, time, 0);
    }

    // **********************************
    // MAIN TIME EVOLUTION LOOP
    // **********************************

    for (int step = 1; step <= nsteps; ++step)
    {
        phi_old.FillBoundary(geom.periodicity());

        for ( amrex::MFIter mfi(phi_old); mfi.isValid(); ++mfi )
        {
            const amrex::Box& bx = mfi.validbox();
            const amrex::Array4<amrex::Real>& phiOld = phi_old.array(mfi);
            const amrex::Array4<amrex::Real>& phiNew = phi_new.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                phiNew(i,j,k) = phiOld(i,j,k) + dt *
                    ( (phiOld(i+1,j,k) - 2.*phiOld(i,j,k) + phiOld(i-1,j,k)) / (dx[0]*dx[0])
                     +(phiOld(i,j+1,k) - 2.*phiOld(i,j,k) + phiOld(i,j-1,k)) / (dx[1]*dx[1])
                     +(phiOld(i,j,k+1) - 2.*phiOld(i,j,k) + phiOld(i,j,k-1)) / (dx[2]*dx[2])
                        );
            });
        }

        time = time + dt;
        amrex::MultiFab::Copy(phi_old, phi_new, 0, 0, 1, 0);

        // Update result with current state
        result.final_step = step;
        result.final_time = time;

        amrex::Print() << "Advanced step " << step << "\n";

        if (plot_int > 0 && step%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",step,5);
            WriteSingleLevelPlotfile(pltfile, phi_new, {"phi"}, geom, time, step);
        }
    }

        // Get final max temperature and mark success
        result.max_temperature = phi_new.max(0);
        result.success = true;

    }
    amrex::Finalize();
    return result;
}

int main(int argc, char* argv[])
{
    heat_equation_main(argc, argv);
    return 0;
}