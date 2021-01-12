#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef __CUDACC__  
#define __CUDACC__ 
#endif

#include <device_functions.h>
#include <Windows.h>
#include <omp.h>
#include <iostream>
#include <assert.h>
#include <iostream>
#include <ctime>

using namespace std;

#define BLOCK_SIZE 8

void inicialization(int arraySize, int* matrix1, int* matrix2, int* resultMatrix)
{
    for (int i = 0; i < arraySize; i++)
    {
        for (int j = 0; j < arraySize; j ++)
        {
            matrix1[i * arraySize + j] = rand() % 10 + 1;
            matrix2[i * arraySize + j] = rand() % 10 + 1;
            resultMatrix[i * arraySize] = 0;
        }  
    }
}

void output(int arraySize, int* matrix) {
    for (int i = 0; i < arraySize; i++)
    {
        for (int j = 0; j < arraySize; j++)
        {
            cout << matrix[i * arraySize + j] << " ";
        }
        cout << endl;
    }
}

void remove(int n, int *matrix)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            matrix[i * n + j] = 0;
        }
}

void serial(int n, int* A, int* B, int* C, int* save)
{
    int stage;
    int* A1, * B1, * C1;
    int n_threads = omp_get_num_threads();
    *save = n_threads;
    int n_blocks = sqrt(n_threads);
    int block_size = n / n_blocks;
    int PrNum = omp_get_thread_num();
    int i1 = PrNum / n_blocks, j1 = PrNum % n_blocks;
    for (stage = 0; stage < n_blocks; ++stage) {
        A1 = A + (i1 * n + ((i1 + stage) % n_blocks)) * block_size;
        B1 = B + (((i1 + stage) % n_blocks) * n + j1) * block_size;
        C1 = C + (i1 * n + j1) * block_size;
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                for (int k = 0; k < block_size; ++k) {
                    *(C1 + i * n + j) += *(A1 + i * n + k) * *(B1 + k * n + j);
                }
            }
        }
    }

}

void multiply_omp(int m_size, int* A, int* B, int* C,  int* save) {
    int stage;
    int* A1, * B1, * C1;
    int n_threads = omp_get_num_threads();
    *save = n_threads;
    int n_blocks = sqrt(n_threads);
    int block_size = m_size / n_blocks;
    int PrNum = omp_get_thread_num();
    int i1 = PrNum / n_blocks, j1 = PrNum % n_blocks;
    for (stage = 0; stage < n_blocks; ++stage) {
        A1 = A + (i1 * m_size + ((i1 + stage) % n_blocks)) * block_size;
        B1 = B + (((i1 + stage) % n_blocks) * m_size + j1) * block_size;
        C1 = C + (i1 * m_size + j1) * block_size;
#pragma omp parallel for
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                for (int k = 0; k < block_size; ++k) {
                    *(C1 + i * m_size + j) += *(A1 + i * m_size + k) * *(B1 + k * m_size + j);
                }
            }
        }
    }
}

__global__ void fox_kernel(int* A, int* B, int* C, int n)
{
    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];
    int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y, Col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int Pvalue = 0;

    if (Row < n && ((n - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE + threadIdx.x < n)
        As[threadIdx.y][threadIdx.x] = A[Row * n + ((n - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE + threadIdx.x];
    else
        As[threadIdx.y][threadIdx.x] = 0;
    if (Col < n && ((n - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE + threadIdx.y < n)
        Bs[threadIdx.y][threadIdx.x] = B[(((n - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE + threadIdx.y) * n + Col];
    else Bs[threadIdx.y][threadIdx.x] = 0;
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; ++k)
        Pvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();

    if(Row < n && Col < n)
        C [Row * n + Col] = Pvalue;
}

int main()
{
    setlocale(LC_ALL, "Russian");
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);

    double start_time, end_time, search_time;
    float KernelTime1, KernelTime2, KernelTime3;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n;
    cout << "Введите размерность матрицы n: " << endl;
    cin >> n;

    int blockSize = BLOCK_SIZE;

    size_t size = n * n * sizeof(int);

    int* matrix1, * matrix2, * result;
    int* device_matrix1, * device_matrix2, * device_result; 

    matrix1 = new int[n * n];
    matrix2 = new int[n * n];
    result = new int[n * n];

    inicialization(n, matrix1, matrix2, result);

    //cout << "\nПервая матрица: " << endl;
    //output(n, matrix1);
    //cout << "\nВторая матрица: " << endl;
    //output(n, matrix2);

    //выделяем память для матриц на GPU (на девайсе)
    cudaMalloc((void**)&device_matrix1, size);
    cudaMalloc((void**)&device_matrix2, size);
    cudaMalloc((void**)&device_result, size);

    //определение размеров сетки и блоков
    dim3 dimGrid = dim3(n / BLOCK_SIZE, n / BLOCK_SIZE);
    dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);

    cout << "\nCUDA:" << endl;
    cudaEventRecord(start, 0);
    cudaMemcpy(device_matrix1, matrix1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix2, matrix2, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, result, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime1, start, stop);
    printf("Время копирования матрицы с хоста на девайс в разделяемую память: %f миллисекунд\n", KernelTime1);

    //умножение матриц в разделяемой памяти
    cudaEventRecord(start, 0);
    fox_kernel << <dimGrid, dimBlock >> > (device_matrix1, device_matrix2, device_result, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime2, start, stop);
    printf("Умножение матриц в разделяемой памяти: %f миллисекунд\n", KernelTime2);
    
    cudaEventRecord(start, 0);
    cudaMemcpy(result, device_result, size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime3, start, stop);
    printf("Время копирование результата с девайса на хост: %f миллисекунд\n", KernelTime3);
    //output(n, result);
    remove(n, result);

    int n_threads = -1;
    start_time = omp_get_wtime();
    multiply_omp(n, matrix1, matrix2, result, &n_threads);
    end_time = omp_get_wtime();
    search_time = end_time - start_time;
    printf("\nУмножение матриц с OMP: %f секунд \n", search_time);
    //output(n, result);
    remove(n, result);

    int s_t = clock();
    serial(n, matrix1, matrix2, result, &n_threads);
    cout << "\nПоследовательное умножение: " << clock() - s_t<< " миллисекунд" << endl;
    //output(n, result);
    remove(n, result);

    free(matrix1);
    free(matrix2);
    free(result);
    cudaFree(device_matrix1);
    cudaFree(device_matrix2);
    cudaFree(device_result);

    return 0;
}
