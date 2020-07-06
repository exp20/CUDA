#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <curand_kernel.h>


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <limits>//
using namespace std;


static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

struct Poly
{
	double Coefficients[5];
	double Error;
	__host__ __device__ Poly()
	{
		for (char i = 0; i < 5; ++i)
		{
			
			//Coefficients[i] = 0.;
			Coefficients[i] = (rand() / double(RAND_MAX) / 10000); /// rand  = 32767
		}
	}

	__host__ __device__ Poly(double c0, double c1, double c2, double c3, double c4) {
		Coefficients[0] = c0;
		Coefficients[1] = c1;
		Coefficients[2] = c2;
		Coefficients[3] = c3;
		Coefficients[4] = c4;
	}

	__host__ __device__ ~Poly()
	{
	}
	
	__host__ __device__ bool operator<(const Poly& other) const
	{
		return this->Error < other.Error;
	}

	static int compare(const void *x1, const void *x2)
	{

		if (((*(Poly*)x1).Error >(*(Poly*)x2).Error))
			return 1;
		if (((*(Poly*)x1).Error < (*(Poly*)x2).Error))
			return -1;
		return 0;
	}
};

double *GetDataFromFile(const char *FileName, int *count)
{
	FILE *Stream;
	if ((Stream = fopen(FileName, "r")) == NULL)
	{
		printf("Failed to open file.");
		return nullptr;
	}
	if (*count == NULL)
		fscanf(Stream, "%i", count);
	double *Point = (double*)malloc(*count * sizeof(double));
	for (int i = 0; i < *count; ++i)
	{
		fscanf(Stream, "%lf", &Point[i]);
	}
	fclose(Stream);
	return Point;
}

void WriteToFile(const char *FileName, double *Source, int count = 5, bool saveCount = false, const char *separator = "\t")
{
	FILE *Stream;
	if ((Stream = fopen(FileName, "w")) == NULL)
	{
		printf("Failed to open file.");
		return;
	}
	if (saveCount)
	{
		fprintf(Stream, "%i", count);
		fprintf(Stream, separator);
	}
	for (int i = 0; i < count - 1; ++i)
	{
		fprintf(Stream, "%.20lf", Source[i]);
		fprintf(Stream, separator);
	}
	fprintf(Stream, "%.20lf", Source[count - 1]);
	fclose(Stream);
}



// ВЫсчит СКО
__global__ void Fitness(double *x, double *y, Poly *individuals, const int numberOfPoints, const int numberOfIndividuals)
{
	const int individual = blockIdx.x * blockDim.x + threadIdx.x;
	if (individual < numberOfIndividuals)
	{
		double mean_sauare_err = 0.;
		long approximatingFunction;
		for (int j = 0; j < numberOfPoints; ++j)
		{
			approximatingFunction = 0.;
			for (int k = 0; k < 5; ++k)
			{
				approximatingFunction += individuals[individual].Coefficients[k] * pow(x[j], (double)k);
			}
			mean_sauare_err += pow(approximatingFunction - y[j], 2);
		}
		individuals[individual].Error = mean_sauare_err;
	}
}

// Селекция
// Оставляем лучшую половину
__global__ void Selection(Poly *individuals, const int numberOfIndividuals, const int threshold)

{
	
	const int individual = blockIdx.x * blockDim.x + threadIdx.x;
	if (individual < numberOfIndividuals- threshold)
	{
		for (char i = 0; i < 5; ++i)	
		{
			individuals[individual+ threshold].Coefficients[i] = individuals[individual].Coefficients[i];
		}
	}
}

// Скрещивание
__global__ void Crossover(Poly *individuals, const int numberOfIndividuals, const int threshold)
{
	
	
	const int individual = blockIdx.x * blockDim.x + threadIdx.x ; // скрещиваем одну половину с другой
	if ((individual < numberOfIndividuals- threshold) && (individual % 2 !=0))
	{
		curandState state;
		double exchange;

		for (int j = 0; j < 5; ++j)	
		{
			curand_init((unsigned long long)clock() + individual, 0, 0, &state);
			if ((curand_uniform(&state) + 3/5) >= 0)	// равномерное распределение от 0 до 1, скрещивание в d в зависимости от вероятности 
			{
				exchange = individuals[individual].Coefficients[j];
				individuals[individual].Coefficients[j] = individuals[individual + 1].Coefficients[j];
				individuals[individual + 1].Coefficients[j] = exchange;
			}
		}
	}

}


// мутируют только дети и та 25% чатсь что осталась от половины родителей
// дети на нечетных элементах второй половины
// так как отсортированные значения то чем больше индекс тем больше ошибка
__global__ void Mutation(Poly *individuals, const int numberOfIndividuals, const int threshold, const double mean, const double variance)
{
	
	
	const int individual = blockIdx.x * blockDim.x + threadIdx.x + threshold ;
	// мутирует 25% из тех что были родителями
    double mutated;
	if (individual < numberOfIndividuals && ((individual % 2)==0))
	{
		curandState state;
		double p = 0.9; ///
		curand_init((unsigned long long)clock() + individual, 0, 0, &state);
		double normal_p = curand_normal_double(&state); // M = 0, D=1
		for (int j = 0; j < 5; ++j)
		{
			
			if ((normal_p + p) >0) //Шанс мутации гена ~50%
			{	
				curand_init((unsigned long long)clock() + individual, 0, 0, &state);
				mutated = curand_normal_double(&state) * variance + mean;
				individuals[individual].Coefficients[j] += mutated;
			}
			else {
				continue;
			}
			
		}
	}
	if (individual < numberOfIndividuals) // дети мутируют
	{
		curandState state;
	
		double p = 0.7; /// 
		for (int j = 0; j < 5; ++j)
		{
			curand_init((unsigned long long)clock() + individual, 0, 0, &state);
			if (curand_normal_double(&state)-p > 0) 
			{
				
				mutated = curand_normal_double(&state) * variance + mean;
				individuals[individual].Coefficients[j] += mutated;
			}

		}
	}
	
}
// В итоге туту родители не мтировали, дети мутировали слегка, 25% часть мутировала в зависимости от ошибки
int main()
{
	
	int numberOfPoints, CurrentEpoch = 0, maxNumberOfEpochs, numberOf_Error_NoChangeEpochs, numberOfIndividuals;
	double *X_real = NULL, *Y_real = NULL, *Random_generation = NULL;
	double meanMutate, varianceMutate, stopErrorThreshold;
	

	Poly *polynomials;
	printf("\nEnter number of individuals: ");
	cin >> numberOfIndividuals;
	
	
	printf("\nEnter number of point in File: ");
	cin >> numberOfPoints;
	X_real = GetDataFromFile("InputX.txt", &numberOfPoints);
	Y_real = GetDataFromFile("InputY.txt", &numberOfPoints);
	if (X_real != NULL && Y_real != NULL) {
		printf("Succes load X_real, Y_real");
	}
	printf("\nLoad random generation from file? 1-Yes, 0-No: ");
	int a;
	cin >> a;
	
	if (a == 1) {
		int numb_gen_points = numberOfIndividuals*5;
		Random_generation = GetDataFromFile("Random.txt", &numb_gen_points); // Рандомные коэффициенты
		polynomials = (Poly*)malloc(numberOfIndividuals * sizeof(Poly));
		for (int i = 0; i < numberOfIndividuals; i++)
		{
			polynomials[i] = Poly(Random_generation[i], Random_generation[i+1], Random_generation[i+2],
				Random_generation[i+3], Random_generation[i+4]);
		}
		for (int i = 0; i < numberOfPoints; i++) {

		}
	}
	else {
		// Инициализация нулевого поколения 
		polynomials = (Poly*)malloc(numberOfIndividuals * sizeof(Poly));
		for (int i = 0; i < numberOfIndividuals; i++)
		{
			polynomials[i] = Poly();
		}
	}
	

	
	cout << "\nMean of mutation: ";
	cin >> meanMutate;
	cout << "\nVariance of mutation: ";
	cin >> varianceMutate;
	cout << "\nNumber of max epochs: ";
	cin >> maxNumberOfEpochs;
	cout << "\nStop threshold error: ";
	cin >> stopErrorThreshold;
	cout << "\nNumber of epochs with no change error: ";
	cin >> numberOf_Error_NoChangeEpochs;



	double *X_GPU, *Y_GPU;
	HANDLE_ERROR(cudaMalloc((void**)&X_GPU, numberOfPoints * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&Y_GPU, numberOfPoints * sizeof(double)));
	HANDLE_ERROR(cudaMemcpy(X_GPU, X_real, numberOfPoints * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(Y_GPU, Y_real, numberOfPoints * sizeof(double), cudaMemcpyHostToDevice));
	free(X_real);
	free(Y_real);
	
	
	
		
	const int threshold = int(numberOfIndividuals / 2.f + 0.5f),
		threadsPerBlockDim = 32,
		blocksPerGridDimX = (const int)ceilf(numberOfIndividuals / (float)threadsPerBlockDim); // кол-во блоков для покрытия всех индивидуумов
	const dim3  gridDim(blocksPerGridDimX, 1, 1), blockDim(threadsPerBlockDim, 1, 1);


		

	Poly *polynomialsOnGPU;
	HANDLE_ERROR(cudaMalloc((void**)&polynomialsOnGPU, numberOfIndividuals * sizeof(Poly)));
	HANDLE_ERROR(cudaMemcpy(polynomialsOnGPU, polynomials, numberOfIndividuals * sizeof(Poly), cudaMemcpyHostToDevice));

	
	free(polynomials);
	// Используется для записи минимального элемента 
	Poly *polynomial = (Poly*)malloc(1 * sizeof(Poly));
	polynomial[0] = Poly();

	// Основной цикл программы
	clock_t startTimer, stopTimer;
	startTimer = clock();
	double lastError = -99;
	for (int i = 0; i < maxNumberOfEpochs; ++i)
	{
		Fitness << <gridDim, blockDim >> > (X_GPU, Y_GPU, polynomialsOnGPU, numberOfIndividuals, numberOfPoints);
		
		//// Использование Thrust sort
		// cортируется на GPU, должен по крайней мере
		thrust::sort(thrust::device, polynomialsOnGPU, polynomialsOnGPU + numberOfIndividuals); // Начало и конец последовательностей
		
		
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaMemcpy(polynomial, polynomialsOnGPU, 1 * sizeof(Poly), cudaMemcpyDeviceToHost));

		int countConstEpocheError = 0;
		
		printf("Epoch %i. Minimal error (e^2) = %lf\n", i, polynomial[0].Error);
		if (polynomial[0].Error <= stopErrorThreshold) {
			cout << "\nReached error treshhold, current error: " << polynomial[0].Error;
		}
		if (i > 0)
		{
			if (lastError <= polynomial[0].Error) { // увеличиваем счетчик эпохи если не увеличиавается точность
				countConstEpocheError += 1;
				
			}
			else {
				countConstEpocheError = 0;
				lastError = polynomial[0].Error;
			}

		}
		if (CurrentEpoch >= maxNumberOfEpochs || countConstEpocheError == numberOf_Error_NoChangeEpochs) {
			cout << "\nReach numberOf_Error_NoChangeEpochs = " << numberOf_Error_NoChangeEpochs;
			break;
		}
		
		// Редактирование ген признаков
		cudaDeviceSynchronize();
		
		Selection << <gridDim, blockDim >> >(polynomialsOnGPU, numberOfIndividuals, threshold);
		
		Crossover << <gridDim, blockDim >> >(polynomialsOnGPU, numberOfIndividuals, threshold);
		
     	Mutation << <gridDim, blockDim >> >(polynomialsOnGPU, numberOfIndividuals, threshold, meanMutate, varianceMutate);
	}
	
	stopTimer = clock();
	printf("Time on GPU = %lf seconds.\n", (double)(stopTimer - startTimer) / CLOCKS_PER_SEC);
	HANDLE_ERROR(cudaMemcpy(polynomial, polynomialsOnGPU, 1 * sizeof(Poly), cudaMemcpyDeviceToHost));
	for (char i = 0; i < 5 - 1; ++i)
	{
		printf("%.20lf * x^%i + ", polynomial[0].Coefficients[i], i);
	}
	printf("%.20lf * x^%i\n", polynomial[0].Coefficients[5 - 1], 5 - 1);
	printf("с0 = %d\n", polynomial[0].Coefficients[5 - 1], 5 - 1);
	printf("blocksPerGridDimX (points) %i.\n", blocksPerGridDimX);
	WriteToFile("Output.txt", polynomial[0].Coefficients);
	HANDLE_ERROR(cudaFree(polynomialsOnGPU));
	HANDLE_ERROR(cudaFree(X_GPU));
	HANDLE_ERROR(cudaFree(Y_GPU));
	free(polynomial);
	int g;
	cin >> g;
}
