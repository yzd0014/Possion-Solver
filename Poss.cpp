#include <iostream>
#include <chrono>
#include <ctime>
#include <vector> 
#include <mpi.h> 
#include <math.h>
#include "omp.h"

#define pi 3.1415926
int main(int argc, char * argv[]) {
	omp_set_dynamic(0);     // Explicitly disable dynamic teams
	omp_set_num_threads(1);
	MPI_Init(&argc, &argv);
	int rank, size;

	//create 3d comm
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int q = std::pow(size, 1.0 / 3.0);
	int trueSize = q * q * q;
	int dims[3] = { q, q, q};
	int periods[3] = { 0, 0, 0 };
	MPI_Comm comm3D;
	MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &comm3D);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int n = 400; //total numbers in 1D
	if (rank < trueSize) {
		int m = n / q;
		int trueN = m * q;
		float h = 1.0f / (trueN - 1);
		m = m + 2;//extra two elements to store halo

		float err = 1.0f;
		float threshold = 0.00001f;

		//initialize local cube
		int localSize = m * m * m;
		float* u = new float[localSize];
		float* u_bar = new float[localSize];
		for (int i = 0; i < localSize; i++) {
			u[i] = 0.0f;
		}

		MPI_Status status;
		int coord[3];
		MPI_Cart_coords(comm3D, rank, 3, coord);

		int counter = 0;
		MPI_Barrier(comm3D);
		double startTime = MPI_Wtime();
		do {
			int surfaceArea = (m - 2) * (m - 2);
			//send down
			MPI_Request req_down;
			float* bottomSurface = nullptr;
			if (coord[0] > 0) {
				int destCoord[3];
				destCoord[0] = coord[0] - 1;
				destCoord[1] = coord[1];
				destCoord[2] = coord[2];

				int dest;
				MPI_Cart_rank(comm3D, destCoord, &dest);

				bottomSurface = new float[surfaceArea];
				int iterator = 0;
				for (int y = 1; y < m - 1; y++) {
					for (int z = 1; z < m - 1; z++) {
						int i = m * m + y * m + z;
						bottomSurface[iterator] = u[i];
						iterator++;
					}
				}
				MPI_Isend(bottomSurface, surfaceArea, MPI_FLOAT, dest, 0, comm3D, &req_down);
			}

			//send up
			MPI_Request req_up;
			float* topSurface = nullptr;
			if (coord[0] < q - 1) {
				int destCoord[3];
				destCoord[0] = coord[0] + 1;
				destCoord[1] = coord[1];
				destCoord[2] = coord[2];

				int dest;
				MPI_Cart_rank(comm3D, destCoord, &dest);

				topSurface = new float[surfaceArea];
				int iterator = 0;
				for (int y = 1; y < m - 1; y++) {
					for (int z = 1; z < m - 1; z++) {
						int i = (m - 2) *m*m + y * m + z;
						topSurface[iterator] = u[i];
						iterator++;
					}
				}
				MPI_Isend(topSurface, surfaceArea, MPI_FLOAT, dest, 0, comm3D, &req_up);
			}
			//send forward
			MPI_Request req_forward;
			float* frontSurface = nullptr;
			if (coord[1] > 0) {
				int destCoord[3];
				destCoord[0] = coord[0];
				destCoord[1] = coord[1] - 1;
				destCoord[2] = coord[2];

				int dest;
				MPI_Cart_rank(comm3D, destCoord, &dest);

				frontSurface = new float[surfaceArea];
				int iterator = 0;
				for (int x = 1; x < m - 1; x++) {
					for (int z = 1; z < m - 1; z++) {
						int i = x * m *m + m + z;
						frontSurface[iterator] = u[i];
						iterator++;
					}
				}
				MPI_Isend(frontSurface, surfaceArea, MPI_FLOAT, dest, 0, comm3D, &req_forward);
			}
			//send backward
			MPI_Request req_backward;
			float* backSurface = nullptr;
			if (coord[1] < q - 1) {
				int destCoord[3];
				destCoord[0] = coord[0];
				destCoord[1] = coord[1] + 1;
				destCoord[2] = coord[2];

				int dest;
				MPI_Cart_rank(comm3D, destCoord, &dest);

				backSurface = new float[surfaceArea];
				int iterator = 0;
				for (int x = 1; x < m - 1; x++) {
					for (int z = 1; z < m - 1; z++) {
						int i = x * m * m + (m - 2) *m + z;
						backSurface[iterator] = u[i];
						iterator++;
					}
				}
				MPI_Isend(backSurface, surfaceArea, MPI_FLOAT, dest, 0, comm3D, &req_backward);
			}
			//send left
			MPI_Request req_left;
			float * leftSurface = nullptr;
			if (coord[2] > 0) {
				int destCoord[3];
				destCoord[0] = coord[0];
				destCoord[1] = coord[1];
				destCoord[2] = coord[2] - 1;

				int dest;
				MPI_Cart_rank(comm3D, destCoord, &dest);

				leftSurface = new float[surfaceArea];
				int iterator = 0;
				for (int x = 1; x < m - 1; x++) {
					for (int y = 1; y < m - 1; y++) {
						int i = x * m * m + y * m + 1;
						leftSurface[iterator] = u[i];
						iterator++;
					}
				}
				MPI_Isend(leftSurface, surfaceArea, MPI_FLOAT, dest, 0, comm3D, &req_left);
			}
			//sent right
			MPI_Request req_right;
			float * rightSurface;
			if (coord[2] < q - 1) {
				int destCoord[3];
				destCoord[0] = coord[0];
				destCoord[1] = coord[1];
				destCoord[2] = coord[2] + 1;

				int dest;
				MPI_Cart_rank(comm3D, destCoord, &dest);

				rightSurface = new float[surfaceArea];
				int iterator = 0;
				for (int x = 1; x < m - 1; x++) {
					for (int y = 1; y < m - 1; y++) {
						int i = x * m * m + y * m + m - 2;
						rightSurface[iterator] = u[i];
						iterator++;
					}
				}
				MPI_Isend(rightSurface, surfaceArea, MPI_FLOAT, dest, 0, comm3D, &req_right);
			}

#pragma omp parallel for			
			for (int x = 2; x < m - 2; x++) {
#pragma omp parallel for
				for (int y = 2; y < m - 2; y++) {
#pragma omp parallel for
					for (int z = 2; z < m - 2; z++) {
						int i = x * m*m + y * m + z;
						int i_down = (x - 1) * m * m + y * m + z;
						int i_up = (x + 1) * m * m + y * m + z;
						int i_forward = x * m * m + (y - 1) * m + z;
						int i_backward = x * m * m + (y + 1) * m + z;
						int i_left = x * m * m + y * m + z - 1;
						int i_right = x * m * m + y * m + z + 1;

						int M = m - 2;
						float X = coord[0] * M + (x - 1);
						float Y = coord[1] * M + (y - 1);
						float Z = coord[2] * M + (z - 1);
						X = X * h;
						Y = Y * h;
						Z = Z * h;
						float f;
						f = -1 * sin(2 * pi * X) * sin(2 * pi * Y) * sin(2 * pi * Y);
						u_bar[i] = (u[i_down] + u[i_up] + u[i_forward] + u[i_backward] + u[i_left] + u[i_right] - h * h * f) / 6.0f;
					}
				}
			}
			
			
			//receive from down
			if (coord[0] > 0) {
				int sourceCoord[3];
				sourceCoord[0] = coord[0] - 1;
				sourceCoord[1] = coord[1];
				sourceCoord[2] = coord[2];

				int source;
				MPI_Cart_rank(comm3D, sourceCoord, &source);
				float * receBuf = new float[surfaceArea];
				MPI_Recv(receBuf, surfaceArea, MPI_FLOAT, source, 0, comm3D, &status);

				int iterator = 0;
				for (int y = 1; y < m - 1; y++) {
					for (int z = 1; z < m - 1; z++) {
						int i = y * m + z;
						u[i] = receBuf[iterator];
						iterator++;
					}
				}
				delete receBuf;
			}
			/*
			if (rank == 4) {
				std::cout << "rank: " << rank << std::endl;
			}*/
			//std::cout << m << std::endl;
			
			//receive from up
			if (coord[0] < q - 1) {
				int sourceCoord[3];
				sourceCoord[0] = coord[0] + 1;
				sourceCoord[1] = coord[1];
				sourceCoord[2] = coord[2];

				int source;
				MPI_Cart_rank(comm3D, sourceCoord, &source);
				float * receBuf = new float[surfaceArea];
				MPI_Recv(receBuf, surfaceArea, MPI_FLOAT, source, 0, comm3D, &status);

				int iterator = 0;
				for (int y = 1; y < m - 1; y++) {
					for (int z = 1; z < m - 1; z++) {
						int i = (m - 1) * m * m + y * m + z;
						u[i] = receBuf[iterator];
						iterator++;
					}
				}
				delete receBuf;
			}
			//recive from forward
			if (coord[1] > 0) {
				int sourceCoord[3];
				sourceCoord[0] = coord[0];
				sourceCoord[1] = coord[1] - 1;
				sourceCoord[2] = coord[2];

				int source;
				MPI_Cart_rank(comm3D, sourceCoord, &source);
				float * receBuf = new float[surfaceArea];
				MPI_Recv(receBuf, surfaceArea, MPI_FLOAT, source, 0, comm3D, &status);

				int iterator = 0;
				for (int x = 1; x < m - 1; x++) {
					for (int z = 1; z < m - 1; z++) {
						int i = x * m * m + z;
						u[i] = receBuf[iterator];
						iterator++;
					}
				}
				delete receBuf;
			}
			//receive from backward
			if (coord[1] < q - 1) {
				int sourceCoord[3];
				sourceCoord[0] = coord[0];
				sourceCoord[1] = coord[1] + 1;
				sourceCoord[2] = coord[2];

				int source;
				MPI_Cart_rank(comm3D, sourceCoord, &source);
				float * receBuf = new float[surfaceArea];
				MPI_Recv(receBuf, surfaceArea, MPI_FLOAT, source, 0, comm3D, &status);

				int iterator = 0;
				for (int x = 1; x < m - 1; x++) {
					for (int z = 1; z < m - 1; z++) {
						int i = x * m * m + (m - 1) * m + z;
						u[i] = receBuf[iterator];
						iterator++;
					}
				}
				delete receBuf;
			}
			//recieve from left
			if (coord[2] > 0) {
				int sourceCoord[3];
				sourceCoord[0] = coord[0];
				sourceCoord[1] = coord[1];
				sourceCoord[2] = coord[2] - 1;

				int source;
				MPI_Cart_rank(comm3D, sourceCoord, &source);
				float * receBuf = new float[surfaceArea];
				MPI_Recv(receBuf, surfaceArea, MPI_FLOAT, source, 0, comm3D, &status);

				int iterator = 0;
				for (int x = 1; x < m - 1; x++) {
					for (int y = 1; y < m - 1; y++) {
						int i = x * m * m + y * m;
						u[i] = receBuf[iterator];
						iterator++;
					}
				}
				delete receBuf;
			}
			//receive from right
			if (coord[2] <  q - 1) {
				int sourceCoord[3];
				sourceCoord[0] = coord[0];
				sourceCoord[1] = coord[1];
				sourceCoord[2] = coord[2] + 1;

				int source;
				MPI_Cart_rank(comm3D, sourceCoord, &source);
				float * receBuf = new float[surfaceArea];
				MPI_Recv(receBuf, surfaceArea, MPI_FLOAT, source, 0, comm3D, &status);

				int iterator = 0;
				for (int x = 1; x < m - 1; x++) {
					for (int y = 1; y < m - 1; y++) {
						int i = x * m * m + y * m + m - 1;
						u[i] = receBuf[iterator];
						iterator++;
					}
				}
				delete receBuf;
			}
#pragma omp parallel for			
			for (int x = 1; x < m - 1; x++) {
#pragma omp parallel for
				for (int y = 1; y < m - 1; y += m - 3) {
#pragma omp parallel for
					for (int z = 1; z < m - 1; z++) {
						int i = x * m*m + y * m + z;
						int i_down = (x - 1) * m * m + y * m + z;
						int i_up = (x + 1) * m * m + y * m + z;
						int i_forward = x * m * m + (y - 1) * m + z;
						int i_backward = x * m * m + (y + 1) * m + z;
						int i_left = x * m * m + y * m + z - 1;
						int i_right = x * m * m + y * m + z + 1;

						int M = m - 2;
						float X = coord[0] * M + (x - 1);
						float Y = coord[1] * M + (y - 1);
						float Z = coord[2] * M + (z - 1);
						X = X * h;
						Y = Y * h;
						Z = Z * h;
						float f;
						f = -1 * sin(2 * pi * X) * sin(2 * pi * Y) * sin(2 * pi * Y);
						u_bar[i] = (u[i_down] + u[i_up] + u[i_forward] + u[i_backward] + u[i_left] + u[i_right] - h * h * f) / 6.0f;
					}
				}
			}
#pragma omp parallel for
			for (int x = 1; x < m - 1; x++) {
#pragma omp parallel for
				for (int y = 2; y < m - 2; y++) {
#pragma omp parallel for
					for (int z = 1; z < m - 1; z += m - 3) {
						int i = x * m*m + y * m + z;
						int i_down = (x - 1) * m * m + y * m + z;
						int i_up = (x + 1) * m * m + y * m + z;
						int i_forward = x * m * m + (y - 1) * m + z;
						int i_backward = x * m * m + (y + 1) * m + z;
						int i_left = x * m * m + y * m + z - 1;
						int i_right = x * m * m + y * m + z + 1;

						int M = m - 2;
						float X = coord[0] * M + (x - 1);
						float Y = coord[1] * M + (y - 1);
						float Z = coord[2] * M + (z - 1);
						X = X * h;
						Y = Y * h;
						Z = Z * h;
						float f;
						f = -1 * sin(2 * pi * X) * sin(2 * pi * Y) * sin(2 * pi * Y);
						u_bar[i] = (u[i_down] + u[i_up] + u[i_forward] + u[i_backward] + u[i_left] + u[i_right] - h * h * f) / 6.0f;
					}
				}
			}
#pragma omp parallel for
			for (int x = 1; x < m - 1; x += m - 3) {
#pragma omp parallel for
				for (int y = 2; y < m - 2; y++) {
#pragma omp parallel for
					for (int z = 2; z < m - 2; z++) {
						int i = x * m*m + y * m + z;
						int i_down = (x - 1) * m * m + y * m + z;
						int i_up = (x + 1) * m * m + y * m + z;
						int i_forward = x * m * m + (y - 1) * m + z;
						int i_backward = x * m * m + (y + 1) * m + z;
						int i_left = x * m * m + y * m + z - 1;
						int i_right = x * m * m + y * m + z + 1;

						int M = m - 2;
						float X = coord[0] * M + (x - 1);
						float Y = coord[1] * M + (y - 1);
						float Z = coord[2] * M + (z - 1);
						X = X * h;
						Y = Y * h;
						Z = Z * h;
						float f;
						f = -1 * sin(2 * pi * X) * sin(2 * pi * Y) * sin(2 * pi * Y);
						u_bar[i] = (u[i_down] + u[i_up] + u[i_forward] + u[i_backward] + u[i_left] + u[i_right] - h * h * f) / 6.0f;
					}
				}
			}

			//calucate loop condition
			err = 0.0f;
			for (int x = 1; x < m - 1; x++) {
				for (int y = 1; y < m - 1; y++) {
					for (int z = 1; z < m - 1; z++) {
						int i = x * m*m + y * m + z;
						err += abs(u[i] - u_bar[i]);
					}
				}
			}
			err = err / pow(m - 2, 3);
			
			//swap
			for (int x = 1; x < m - 1; x++) {
				for (int y = 1; y < m - 1; y++) {
					for (int z = 1; z < m - 1; z++) {
						int i = x * m*m + y * m + z;
						u[i] = u_bar[i];
					}
				}
			}
			
			if (coord[0] > 0) {
				MPI_Wait(&req_down, &status);
				delete bottomSurface;
			}
			if (coord[0] < q - 1) {
				MPI_Wait(&req_up, &status);
				delete topSurface;
			}
			if (coord[1] > 0) {
				MPI_Wait(&req_forward, &status);
				delete frontSurface;
			}
			if (coord[1] < q - 1) {
				MPI_Wait(&req_backward, &status);
				delete backSurface;
			}

			if (coord[2] > 0) {
				MPI_Wait(&req_left, &status);
				delete leftSurface;
			}
			if (coord[2] < q - 1) {
				MPI_Wait(&req_right, &status);
				delete rightSurface;
			}
			counter++;
			MPI_Bcast(&err, 1, MPI_FLOAT, 0, comm3D);

		} while (err > threshold);
		
		MPI_Barrier(comm3D);
		double endTime = MPI_Wtime();
		if (rank == 0) {
			double elapsedTime = endTime - startTime;
			std::cout << "Time: " << elapsedTime << std::endl;
		}

		delete u_bar;
		delete u;
	}

	MPI_Finalize();
	
	return 0;
}

