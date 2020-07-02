#include <malloc.h>
#include <stdio.h>
//#include "device_launch_parameters.h" ///  Теперь находятся идентификаторы CUDA
#include <string>
#include <curand.h>
#include <cstdlib>
#include <ctime>

using namespace std;


__global__ void KernelCodeMatMul(int *A, int *B, int *C, int N)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y; // вычисляем номер строки и номер столбца элемента соответствующий номеру текущей нити
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < N && col < N)
	{
		int Sum = 0;

		if (row < N && col < N) {
			// чтоб не выйти за границы если указаны не верные размеры
			for (int i = 0; i < N; i++) {
				Sum += A[row * N + i] * B[i * N + col];
			}
		}
		C[row * N + col] = Sum;
	}
}

int main(int argc, char* argv[]) {
	/// Реализовано умножение квадратных матриц
	int N = atoi(argv[1]);
	int SIZE = N * N;
	printf("Size %d\n", N);
	// Выделение памяти на хосту для хранения далее сгенерированных матриц
	// Матрицы хранятся в виде одномерных векторов размером NxN
	int *A = new int[SIZE];
	int *B = new int[SIZE];
	int *C = new int[SIZE];  //сюды записывается результат

	// заполняем матрицы рандомными числами
	for (int i = 0; i < SIZE; i++)
	{
		A[i] = rand();
		B[i] = rand();
	}
	
	if (Flag == 0){
		clock_t startTime = clock();
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < N; ++k)
				{
					C[i * N + j] += A[i * N + k] * B[k * N + j];
				}
			
		}
	}
		clock_t endTime = clock();

		clock_t clockTicksTaken = endTime - startTime;
		double timeInSeconds = clockTicksTaken / (double)CLOCKS_PER_SEC;
		cout<<"\nclockTicksTaken "<< timeInSeconds <<endl;	
	}
		
	if (Flag == 1){
	// Выделение памяти на device (GPU) для сгенерированных матриц
	int *d_A, *d_B, *d_C;
	cudaError_t  err_flag1 = cudaMalloc(&d_A, SIZE * sizeof(int));
	cudaError_t  err_flag2 = cudaMalloc(&d_B, SIZE * sizeof(int));
	cudaError_t  err_flag3 = cudaMalloc(&d_C, SIZE * sizeof(int));

	if (err_flag1 != cudaSuccess || err_flag2 != cudaSuccess || err_flag3 != cudaSuccess) {
		printf("CUDA ERROR In malloc");
		printf("\n%s", cudaGetErrorString(err_flag1));
		printf("\n%s", cudaGetErrorString(err_flag2));
		printf("\n%s", cudaGetErrorString(err_flag3));
	}
	

	//Перекинем сгенерированные днанные с хоста на gpu
	err_flag1 = cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
	err_flag2 = cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

	if (err_flag1 != cudaSuccess || err_flag2 != cudaSuccess) {
		printf("CUDA ERROR In Memcpy");
		printf("\n%s", cudaGetErrorString(err_flag1));
		printf("\n%s", cudaGetErrorString(err_flag2));
	}

	cudaEvent_t start, stop;  // События для вычисления времени вычислений
	float gpuTime = 0.0f;
	err_flag1 = cudaEventCreate(&start);
	if (err_flag1 != cudaSuccess)
	{
		fprintf(stderr, "Cannot create CUDA start event: %s\n",
			cudaGetErrorString(err_flag1));
		return 0;
	}

	err_flag1 = cudaEventCreate(&stop);
	if (err_flag1 != cudaSuccess)
	{
		fprintf(stderr, "Cannot create CUDA end event: %s\n",
			cudaGetErrorString(err_flag1));
		return 0;
	}

	int wurp_size = 32; // Константа установленная разработчиками - размер варпа. Выберем 32*32 в качеств еразмер а блока
	dim3 blockDim(wurp_size, wurp_size,1); // размер блока 
	int blocksPerGridDimX = ceilf(N / wurp_size); //Функции округляют аргумент x до наименьшего целого числа, которое больше или равно аргументу.
	int blocksPerGridDimY = ceilf(N / wurp_size); // квадратная Grid
	dim3 gridDim(blocksPerGridDimX, blocksPerGridDimY,1);
	printf("FFFFF: %d\n\n", blocksPerGridDimX);

	KernelCodeMatMul <<< gridDim, blockDim >>>(d_A, d_B, d_C, N);

	err_flag1 = cudaGetLastError();
	if (err_flag1 != cudaSuccess)
	{
		fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
			cudaGetErrorString(err_flag1));
		return 0;
	}
	err_flag1 = cudaDeviceSynchronize(); // Поскольку некоторые операции могут выполняться асинхронно вызывается cudaEventSynchronize(), что бы метка записалась корректно.
	if (err_flag1 != cudaSuccess)
	{
		fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
			cudaGetErrorString(err_flag1));
		return 0;
	}
	err_flag1 = cudaEventRecord(stop, 0); // stop - запись события, 0 - поток

	if (err_flag1 != cudaSuccess)
	{
		fprintf(stderr, "Cannot cudaEventRecord: %s\n",
			cudaGetErrorString(err_flag1));
		return 0;
	}

	err_flag1 = cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	if (err_flag1 != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy c array from device to host: %s\n",
			cudaGetErrorString(err_flag1));
		return 0;
	}

	err_flag1 = cudaEventElapsedTime(&gpuTime, start, stop);
	printf("time spent executing %s: %.9f seconds\n", "kernel", gpuTime / 1000);



	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(A);
	free(B);
	free(C);
	return 0;
	}
}
