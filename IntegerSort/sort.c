#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <xmmintrin.h>
#include <emmintrin.h>
//#include <immintrin.h>
#include <math.h>
#include <omp.h>
#define BITS 32
void RadixSort_int(int* d,int N);
void prefix_sum_omp_sse(const double *a, double *s, int n);

/*
Initialize integer array d[N] with seed
*/
void sort_gen(int *d,int N,int seed){
    srand(seed);
    for(int i=0;i<N;i++){
        d[i]=rand();
    }
}

int main(int argc, char  *argv[]) {
    int N;
    N = atoi(argv[1]);
    int seed;
    seed = atof(argv[2]);

    struct timeval start;
    struct timeval end;
    float diff;

    //待测试的程序段
    int *d = malloc(sizeof(int) * N);

    sort_gen(d,N,seed);
    RadixSort_int(d, N);
    printf("%d\n",d[N/2]);
    return 0;
}

void RadixSort_int(int* d,int N)
{
    int D=9;
    int radix = 1<<D;
    int* result = malloc(sizeof(int) * N);
    int* histogram =malloc(sizeof(int) * (radix));
    int* offset =malloc(sizeof(int) * (radix));

    for (int bit_pos = 0;bit_pos<BITS; bit_pos+=D ) {

        #pragma omp parallel for schedule(static)
        for (int i = 0;i<radix+1;i++){
            histogram[i]=0;
            offset[i]=0;
        }

        for (int i = 0; i < N; i++ ) {
            histogram[(d[i]>>bit_pos & (radix-1))]++;
        }

        for (int i = 0;i<radix+1;i++)
        {
            offset[i] = offset[i-1] + histogram[i-1];
        }

        for (int i = 0; i < N; i++ ) {
            result[(int)offset[(d[i]>>bit_pos & (radix-1))]++]=d[i];
        }

        memcpy(d,result,sizeof(int)*N);
    }
}

void prefix_sum_omp_sse(const double* a, double* s, int n) {
    double *suma;
#pragma omp parallel
    {
        const int ithread = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
#pragma omp single
        {
            suma = malloc(sizeof(double) * (nthreads+1));
            suma[0] = 0;
        }
        double sum = 0;
#pragma omp for schedule(static) nowait
        for (int i = 0; i<n; i++) {
            sum += a[i];
            s[i] = sum;
        }
        suma[ithread + 1] = sum;
#pragma omp barrier
#pragma omp single
        {
            double tmp = 0;
            for (int i = 0; i<(nthreads + 1); i++) {
                tmp += suma[i];
                suma[i] = tmp;
            }
        }
        __m128d offset = _mm_set1_pd(suma[ithread]);
#pragma omp for schedule(static)
        for (int i = 0; i<n/4; i++) {
            __m128d tmp1 = _mm_load_pd(&s[4*i]);
            tmp1 = _mm_add_pd(tmp1, offset);
            __m128d tmp2 = _mm_load_pd(&s[4*i+2]);
            tmp2 = _mm_add_pd(tmp2, offset);
            _mm_store_pd(&s[4*i], tmp1);
            _mm_store_pd(&s[4*i+2], tmp2);
        }
    }
//    free(suma);
}
