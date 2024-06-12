
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <mpi.h>
#include <AMReX_MPMD.H>

int main(int argc, char* argv[])
{
    // Initialize amrex::MPMD to establish communication across the two apps
    MPI_Comm comm = amrex::MPMD::Initialize(argc, argv);
    amrex::Initialize(argc,argv,true,comm);
    {
        amrex::Print() << "Hello world from AMReX version " << amrex::Version() << "\n";
    int rank_offset = amrex::MPMD::MyProc() - amrex::ParallelDescriptor::MyProc();
    int this_root, other_root;
    if (rank_offset == 0) { // First program
        this_root = 0;
        other_root = amrex::ParallelDescriptor::NProcs();
    } else {
        this_root = rank_offset;
        other_root = 0;
    }

    std::cout<<"My rank is "<<amrex::MPMD::MyProc()<<" out of "<<amrex::MPMD::NProcs()<<" total ranks in MPI_COMM_WORLD communicator "<<MPI_COMM_WORLD<< "and my rank is "<<amrex::ParallelDescriptor::MyProc()<<" out of "<<amrex::ParallelDescriptor::NProcs()<<" total ranks in my part of the split communicator for the appnum (color) "<< amrex::MPMD::AppNum()<<std::endl;

    int this_nboxes = 10; // setting this arbitrarily
    int other_nboxes = this_nboxes;

    // BOTH SEND AND RECV
    if (amrex::MPMD::MyProc() == this_root) {
        if (rank_offset == 0) // the first program
        {
            MPI_Send(&this_nboxes, 1, MPI_INT, other_root, 0, MPI_COMM_WORLD);
            MPI_Recv(&other_nboxes, 1, MPI_INT, other_root, 1, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
        else // the second program
        {
	    MPI_Recv(&other_nboxes, 1, MPI_INT, other_root, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Send(&this_nboxes, 1, MPI_INT, other_root, 1, MPI_COMM_WORLD);
        }
    }
    this_nboxes*=2;
    amrex::Print()<<"Recieved other_nboxes as "<<other_nboxes<<" sending again after this_nboxes*=2=\t"<<this_nboxes<<std::endl;
    // JUST SEND
    if (amrex::MPMD::MyProc() == this_root) {
        if (rank_offset == 0) // the first program
        {
            MPI_Send(&this_nboxes, 1, MPI_INT, other_root, 0, MPI_COMM_WORLD);
        }
        else // the second program
        {
            MPI_Send(&this_nboxes, 1, MPI_INT, other_root, 1, MPI_COMM_WORLD);
        }
    }

    //JUST RECV
    if (amrex::MPMD::MyProc() == this_root) {
        if (rank_offset == 0) // the first program
        {
            MPI_Recv(&other_nboxes, 1, MPI_INT, other_root, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else // the second program
        {
            MPI_Recv(&other_nboxes, 1, MPI_INT, other_root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    amrex::Print()<<"Recieved other_nboxes as "<<other_nboxes<<std::endl;
    }
    amrex::Finalize();
    amrex::MPMD::Finalize();
}
