#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <xmmintrin.h>
#include <emmintrin.h>
//#include <immintrin.h>
#include <math.h>
#include <omp.h>

int GetDigitInPos(int num,int pos);
void RadixSort(int* d, int N);
void RadixSort_offset(int* d,int N);
void RadixSort_int(int* d,int N);
void RadixSort_double(int* d,int N);
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

int compare_ints(const void* a, const void* b)
{
    int arg1 = *(const int*)a;
    int arg2 = *(const int*)b;

    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;

    // return (arg1 > arg2) - (arg1 < arg2); // possible shortcut
    // return arg1 - arg2; // erroneous shortcut (fails if INT_MIN is present)
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
    printf("%d\n",N);
    printf("%d\n",seed);

    int *d = malloc(sizeof(int) * N);
    printf("%d\n",N/2);
//    int *result = static_cast<int *>(malloc(sizeof(int) * N));

//    sort_gen(d,N,seed);
//    printf("\nRunning quicksort\n");
//    gettimeofday(&start,NULL);
//    qsort(d, N, sizeof(int), compare_ints);
//    printf("%d\n",d[N/2]);
//    gettimeofday(&end,NULL);
//    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
//    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位

//    sort_gen(d,N,seed);
//    printf("\nRunning RadixSort\n");
//    gettimeofday(&start,NULL);
//    RadixSort(d, N);
//    printf("%d\n",d[N/2]);
//    gettimeofday(&end,NULL);
//    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
//    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位

    sort_gen(d,N,seed);
    printf("\nRunning RadixSort int\n");
    gettimeofday(&start,NULL);
    RadixSort_int(d, N);
    printf("%d\n",d[N/2]);
    gettimeofday(&end,NULL);
    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位

    sort_gen(d,N,seed);
    printf("\nRunning RadixSort double\n");
    gettimeofday(&start,NULL);
    RadixSort_double(d, N);
    printf("%d\n",d[N/2]);
    gettimeofday(&end,NULL);
    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位


    return 0;
}


int GetDigitInPos(int num,int pos)
{
    int temp = 1;
    temp = (int)pow(10.0,(double)pos);
    return (num / temp) % 10;
}


void RadixSort(int* d, int N)
{
    int *radixArrays[10];
    for (int i=0; i<10; i++)
    {
        radixArrays[i] = (int *)malloc(sizeof(int)*(N + 1));
        radixArrays[i][0] = 0;
    }
    for (int pos=1; pos< 11; pos++)
    {
        for (int i=0; i<N; i++)
        {
            int num = GetDigitInPos(d[i],pos);
            int index = ++radixArrays[num][0];
            radixArrays[num][index] = d[i];
        }

        for (int i=0, j=0; i<10; i++)
        {
            for (int k = 1; k <= radixArrays[i][0]; k++)
                d[j++] = radixArrays[i][k];
            radixArrays[i][0] = 0;
        }
    }
}

void RadixSort_offset(int* d,int N)
{
#define BITS 32
#define Length 10
    unsigned index0,index1;

    int nzeros;
    int nones;
    int* result = malloc(sizeof(int) * N);
    int* temp;
    for (int bit_pos = 0; bit_pos < BITS; bit_pos++ ) {

        nzeros=0;
#pragma omp parallel for reduction(+:nzeros)
        for (int i = 0; i < N; i++ ) {
            if( !(d[i] & 1<<bit_pos)){
                nzeros++;
            }
        }


        index0 = 0;
        index1 = nzeros;

        for (int i = 0; i < N; i++ ) {
            if ( !(d[i]  & (1<<bit_pos))) {
                result[index0++] = d [i];
            } else {
                result[index1++] = d [i];
            }
        }

//        temp=d;
//        d = result;
//        result =temp;

        memcpy(d,result,sizeof(int)*N);
    }
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

void RadixSort_double(int* d,int N)
{
    int D=9;
    int radix = 1<<D;
    int* result = malloc(sizeof(int) * N);
    double* histogram =malloc(sizeof(double) * (radix+1));
    double* offset =malloc(sizeof(double) * (radix+1));

    for (int bit_pos = 0;bit_pos<BITS; bit_pos+=D ) {

#pragma omp parallel for schedule(static)
        for (int i = 0;i<radix+1;i++){
            histogram[i]=0;
            offset[i]=0;
        }

        for (int i = 0; i < N; i++ ) {
            histogram[(d[i]>>bit_pos & (radix-1))+1]++;
        }

        prefix_sum_omp_sse(histogram,offset,radix);

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

void RadixSort_backup(int* d,int N)
{
    int D=10;
    int B=16;
    int radix = 1<<D;
    int* result = malloc(sizeof(int) * N);
//    double* histogram;
    double* histogram =malloc(sizeof(double) * (radix+1));
    double* offset =malloc(sizeof(double) * (radix+1));
    int* buffer = malloc(sizeof(int)*radix*B);
    int* bufferoffset =malloc(sizeof(int) * radix);

    for (int bit_pos = 0;bit_pos<BITS; bit_pos+=D ) {
#pragma omp for schedule(static)
        for (int i = 0;i<radix+1;i++){
            histogram[i]=0;
            offset[i]=0;
        }

//        for (int i = 0;i<radix;i++){
//            bufferoffset[i]=0;
//        }
//        #pragma omp for schedule(static)
//        for (int i = 0;i<radix*B;i++){
//            buffer[i]=0;
//        }
//        const int nthreads = omp_get_num_threads();
//        #pragma omp single
//        {
//            histogram = malloc(sizeof(double) * nthreads*(radix+1));
//        }
//        #pragma omp for schedule(static)
#pragma omp for schedule(static)
        for (int i = 0; i < N; i++ ) {
//            int ithread = omp_get_thread_num();
            histogram[(d[i]>>bit_pos & (radix-1))+1]++;
        }
//#pragma omp single
//        {
//            for(int i = 0;i<n)
//        }

        prefix_sum_omp_sse(histogram,offset,radix);


//        printf("\ti\thistogram\toffset\t\n");
//        for (int i = 0;i<radix+1;i++)
//        {
////            offset[i] = offset[i-1] + histogram[i-1];
//            printf("\t%d\t%d\t%d\t\n",i,(int)histogram[i],(int)offset[i]);
//        }
//#pragma omp for schedule(static)
//        for (int i = 0; i < N; i++ ) {
//            result[(int)offset[(d[i]>>bit_pos & (radix-1))]++]=d[i];
//        }


        for (int i = 0; i < N; i++ ) {
            int r = d[i]>>bit_pos & 255;
            buffer[r*B+bufferoffset[r]++]=d[i];
            if(bufferoffset[r]==B){
                memmove(result+(int)offset[r],buffer+r*B,sizeof(int)*B);
                for(int j=0;j<B;j++)
                    buffer[r*B+j]=0;
                bufferoffset[r]=0;
                offset[r]+=B;
            }
        }

        memcpy(d,result,sizeof(int)*N);
    };


}
