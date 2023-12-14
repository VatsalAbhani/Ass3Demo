#include <stdio.h>
#define TILE_SIZE 16 

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {
    int bIdx = blockIdx.x;
    int bIdy = blockIdx.y;

    int tIdx = threadIdx.x;
    int tIdy = threadIdx.y;

    int Row = bIdy * blockDim.y + tIdy;
    int Col = bIdx * blockDim.x + tIdx;

   __shared__ float As[TILE_SIZE][TILE_SIZE];
   __shared__ float Bs[TILE_SIZE][TILE_SIZE]; 
 
    float Result = 0.0;

    for(int ph = 0; ph < (k-1) / TILE_SIZE + 1; ++ph) {
	int globalRow = ph * TILE_SIZE + tIdy;
        int globalCol = ph * TILE_SIZE + tIdx;

	if (Row < m && globalCol < k)
	As[tIdy][tIdx] = A[Row * k + globalCol]; 
        else
	As[tIdy][tIdx] = 0.0; 

	if(globalRow < k && Col < n)
	Bs[tIdy][tIdx] = B[globalRow * n + Col];
        else
	Bs[tIdy][tIdx] = 0.0; 

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) 
		Result += As[tIdy][i] * Bs[i][tIdx];
        __syncthreads();
    }

   if(Row < m && Col < n)
    C[Row * n + Col] = Result; 
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    //INSERT CODE HERE
   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n - 1) / dimBlock.x + 1, (m - 1) / dimBlock.y + 1);
    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
	 mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);
    /*************************************************************************/
}



