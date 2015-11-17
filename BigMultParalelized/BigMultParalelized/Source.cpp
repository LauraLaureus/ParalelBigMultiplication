#include <cstdio>
#include <random>

#include <mkl.h>
#include <omp.h>

#define Nbig 4000
#define MAXTHREADS 20

using namespace std;

//Balance workload
void BalanceoCarga(int Nth, int M, int *Pos, int* Num);

void BalanceoCarga(int Nth, int M, int *Pos, int* Num){
	//less tasks than threads
	if (M <= Nth){
		for (int i = 0; i < Nth; i++) Pos[i] = i;
		for (int i = 0; i < M; i++) Num[i] = 1;
		for (int i = M; i < Nth; i++) Num[i] = 0;
		return;
	}
	//more task than threads
	int a = M / Nth;
	for (int i = 0; i < Nth; i++) Num[i] = a;
	int b = M - a*Nth;
	if (b > 0)
		for (int i = 0; i < b; i++) Num[i]++;
	Pos[0]=0;
	for (int i = 1; i < Nth; i++) Pos[i] = Pos[i-1] +Num[i-1] ;
}


int main(int argc, char* argv[]){

	printf("\nUsage of OpenMP and BLAS for Big Matrix multiplication\nAuthor:Laura del Pino Diaz\n");

	int Pos[MAXTHREADS]; //Matrix position for thread
	int Num[MAXTHREADS]; //Number of task thread has to do.
	int Nth, N;
	double *A, *B, *C;

	bool test = false; // set to false to execute real multiplication otherwise a test will run.
	if (test){
		printf("\n Test case\n\n");
			N = 5;
		A = (double*)mkl_malloc(N*N*sizeof(double), 64);
		B = (double*)mkl_malloc(N*N*sizeof(double), 64);
		C = (double*)mkl_malloc(N*N*sizeof(double), 64);

		for (int i = 0; i < N*N; i++)
		{
			A[i] = 1.0 + (double)i;
			B[i] = (double)(N*N) - (double)i;
		}

		Nth = 3;
		BalanceoCarga(Nth, N, Pos, Num);
	}
	else{
		N = Nbig;
		A = (double*)mkl_malloc(N*N*sizeof(double), 64);
		B = (double*)mkl_malloc(N*N*sizeof(double), 64);
		C = (double*)mkl_malloc(N*N*sizeof(double), 64);

		default_random_engine generator;
		normal_distribution<double> dist(0.0, 1.0);

		for (int i = 0; i < N*N; i++)
		{
			A[i] = dist(generator);
			B[i] = dist(generator);
		}

		Nth = 10;
		BalanceoCarga(Nth, N, Pos, Num);
	}
	
	/*Parallelization code*/
	int i;
	printf("Num de hilos: %d", Nth);
	double inicio = omp_get_wtime();
#pragma omp parallel for private(i) num_threads(Nth)
	for (i = 0; i < Nth; i++){
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Num[i], N, N, 1.0, &(A[Pos[i] * N]), N, B, N, 0.0, &(C[Pos[i] * N]), N);
	}
	double fin = omp_get_wtime();


	/*Result Report*/
	if (test){
		for (int a = 0; a < N; a++){
			for (int b = 0; b < N; b++)	printf("%g ", C[a*N + b]);
			printf("\n");
		}
	}
	else{
	
		double tiempo = fin - inicio;
		double Gflops = 2.0*N*N*N*1.0e-09 / tiempo;
		printf("\nNthread: %d, Time: %g s, Gflops: %g\n", Nth, tiempo, Gflops);
	}

	mkl_free(A);
	mkl_free(B);
	getchar();
	return 0;
}