#include <iostream>
#include <mpi.h>
#include <ostream>
#include <string>

double lookupVal(int n, double *x, double *y, double xval) {
  // Lookup
  for (int i = 0; i < n; ++i)
    if (xval >= x[i] and xval <= x[i + 1]) {
      return y[i] + (xval - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    }
  // Return -1 if no result, don't want to return null to non-void function
  return -1;
}

int main(int argc, char *argv[]) {
  // Argument parsing
  if (argc < 2) {
    std::cout << "Missing arguments. Usage: ./lookup <numPE>" << std::endl;
    return -1;
  }
  int numPE = std::stoi(argv[1]);
  int n = 10000;

  // MPI Setup
  MPI_Init(&argc, &argv);
  int myPE;
  int root_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &numPE);
  MPI_Comm_rank(MPI_COMM_WORLD, &myPE);

  // Global variables for all processes
  double x[n], y[n];
  int num_to_lookup = 102;
  double xvals[num_to_lookup];
  if (myPE == 0) {
    // Print diagnostic info
    std::cerr << "Running with array size " << n << " on " << numPE << " cores"
              << std::endl;

    // Fill lookup arrays
    for (int i = 0; i < n; ++i) {
      x[i] = i;
      y[i] = i * i;
    }

    // Generate arbitrary values to lookup
    for (int i = 0; i < num_to_lookup; ++i) {
      xvals[i] = i * (n / num_to_lookup);
    }
  }
  // Broadcast to all processes
  MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(y, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Generate counts/displacements for Scatterv
  int counts[numPE];
  int displacements[numPE];
  // Build array on a single process (maybe unnecessary, small array...)
  if (myPE == 0) {
    int elements_per_process = (int)num_to_lookup / numPE;
    int extra_elements= num_to_lookup % numPE;

    int sum_of_elements = 0;

    for (int i = 0; i < numPE; ++i) {

      counts[i] = elements_per_process;

      // Add spare elements to count if necessary, spreading them as 
      // evenly as possible across processes
      if(i < extra_elements) {
        counts[i]++;
      }

      // Set displacements to sum of elements so far
      displacements[i] = sum_of_elements;

      sum_of_elements += counts[i];
    }
  }

  // Tell each process how many elements they will lookup
  int num_to_process;
  MPI_Scatter(counts, 1, MPI_INT, &num_to_process, 1, MPI_INT, 0, MPI_COMM_WORLD);



  // Scatter elements to lookup to processes with different 
  double process_x_vals[num_to_process];
  MPI_Scatterv(xvals, counts, displacements, MPI_DOUBLE, process_x_vals,
               num_to_process, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);

  // Lookup and print
  double process_y_vals[num_to_process];
  for (int i = 0; i < num_to_process; ++i) {
    double curr_xval = process_x_vals[i];
    process_y_vals[i] = lookupVal(n, x, y, curr_xval);
  }

  double out_x_vals[num_to_lookup]; // Duplicating to keep order...
  double out_y_vals[num_to_lookup];
  MPI_Gatherv(process_x_vals, num_to_process, MPI_DOUBLE, out_x_vals, counts,
              displacements, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
  MPI_Gatherv(process_y_vals, num_to_process, MPI_DOUBLE, out_y_vals, counts,
              displacements, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);


  if (myPE == 0) {
    for (int i = 0; i < num_to_lookup; ++i) {
      std::cout << "(" << i << ") " << out_x_vals[i] << " -> " << out_y_vals[i] << std::endl;
    }
  }

  // End Parallel
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}
