#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <omp.h>

float rand_float(float s);
void matrix_gen(float *a,float *b,int N,float seed);
void matrix_mul(float *a,float *b,float *c,int N);
float compute_trace(float *c, int N);
float get_min(float *c,int N);
void matrix_mul_mp(float *a,float *b,float *c,int N);
void matrix_mul_block(float *a,float *b,float *c,int N);
void matrix_mul_sse(float *a,float *b,float *c,int N);
void matrix_mul_avx(float *a,float *b,float *c,int N);
void matrixMultiplyWithSSE(float* a, float * b, float* result, int N);
void matrixMultiplyOMPAccerate(float* a, float* b, float* result, int N);
void matrixMultiplyBaseLine(float* a, float* b, float* result, int N);
void matrix_mul_sse_block(float *a,float *b,float *c,int N);
void matrix_mul_test(float *a,float *b,float *c,int N);
float get_min_avx(float *c,int N);

int main(int argc, char  *argv[]) {
    int N;
    N = atoi(argv[1]);
    float seed;
    seed = atof(argv[2]);
    float *a = malloc(sizeof(float) *N*N);
    float *b = malloc(sizeof(float) *N*N);
    float *c = malloc(sizeof(float) *N*N);

    struct timeval start;
    struct timeval end;
    float diff,trace;

    //待测试的程序段
    printf("%d\n",N);
    printf("%f\n",seed);
    matrix_gen(a,b,N,seed);

//    printf("\nRunning pure matrix multiply\n");
//    gettimeofday(&start,NULL);
//    matrix_mul(a,b,c,N);
//    trace = compute_trace(c,N);
//    printf("%f\n",trace);
//    gettimeofday(&end,NULL);
//    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
//    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位

//    printf("\nRunning matrix multiply with openMP\n");
//    gettimeofday(&start,NULL);
//    matrix_mul_mp(a,b,c,N);
//    trace = compute_trace(c,N);
//    printf("%f\n",trace);
//    gettimeofday(&end,NULL);
//    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
//    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位

//    printf("\nRunning matrix multiply with block\n");
//    gettimeofday(&start,NULL);
//    matrix_mul_block(a,b,c,N);
//    trace = compute_trace(c,N);
//    printf("%f\n",trace);
//    gettimeofday(&end,NULL);
//    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
//    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位

//    printf("\nRunning matrix multiply with sse\n");
//    gettimeofday(&start,NULL);
//    matrix_mul_sse_block(a,b,c,N);
//    trace = compute_trace(c,N);
//    printf("%f\n",trace);
//    gettimeofday(&end,NULL);
//    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
//    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位

    printf("\nRunning matrix multiply\n");
    gettimeofday(&start,NULL);
    matrix_mul_avx(a,b,c,N);
    trace = get_min(c,N);
    printf("%f\n",trace);
    gettimeofday(&end,NULL);
    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位

    printf("\nRunning matrix multiply with avx\n");
    gettimeofday(&start,NULL);
    matrix_mul_avx(a,b,c,N);
    trace = get_min_avx(c,N);
    printf("%f\n",trace);
    gettimeofday(&end,NULL);
    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位

//    printf("\nRunning matrix multiply with test\n");
//    gettimeofday(&start,NULL);
//    matrix_mul_test(a,b,c,N);
//    trace = get_min(c,N);
//    printf("%f\n",trace);
//    gettimeofday(&end,NULL);
//    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
//    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位
//    return 0;
}

/*
Input: a, b are the N*N float matrix, 0<seed<1, float
        This function should initialize two matrixs with rand_float()
*/
float rand_float(float s){
    return 4*s*(1-s);
}


void matrix_gen(float *a,float *b,int N,float seed){
    float s=seed;
    for(int i=0;i<N*N;i++){
        s=rand_float(s);
        a[i]=s;
        s=rand_float(s);
        b[i]=s;
    }
}

void matrix_mul(float *a,float *b,float *c,int N) {

    for(int i=0;i<N*N;i++)
        c[i]=0.0;
//one-dem
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int pos = i * N + j;
            for (int k = 0; k < N; k++) {
                c[pos] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}

void matrix_mul_mp(float *a,float *b,float *c,int N) {

#pragma omp parallel for num_threads(4)
    for(int i=0;i<N*N;i++)
        c[i]=0.0;
#pragma omp barrier

#pragma omp parallel for num_threads(4)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int pos = i * N + j;
            for (int k = 0; k < N; k++) {
                c[pos] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}


void matrix_mul_block(float *a,float *b,float *c,int N){
    int bsize = 64;
#pragma omp parallel for
    for(int i=0;i<N*N;i++)
        c[i]=0.0;
#pragma omp barrier


    for(int bi=0; bi<N; bi+=bsize){
        for(int bj=0; bj<N; bj+=bsize){
#pragma omp parallel for num_threads(2)
            for(int bk=0; bk<N; bk++){
                for(int j=bj; j<bj+bsize; j++){
                    float sum=0.0;
                    for (int i=bi; i<bi+bsize; i++){
                        sum+=a[bk*N+i]*b[i*N+j];
                    }
                    c[bk*N+j]+=sum;
                }
            }
        }
    }
}
void matrix_mul_avx(float *a,float *b,float *c,int N){
    int bsize = 8;
#pragma omp parallel for
    for(int i=0;i<N*N;i++)
        c[i]=0.0;
#pragma omp barrier

#pragma omp parallel for
    for(int bi=0; bi<N; bi+=bsize){
        for(int bj=0; bj<N; bj+=bsize){
            for(int bk=0; bk<N; bk+=bsize){
                for(int r = bi;r<bi+bsize;r++)
                {
                    __m256 b_1 = _mm256_loadu_ps(b+bj*N+bk);
                    __m256 b_2 = _mm256_loadu_ps(b+(bj+1)*N+bk);
                    __m256 b_3 = _mm256_loadu_ps(b+(bj+2)*N+bk);
                    __m256 b_4 = _mm256_loadu_ps(b+(bj+3)*N+bk);
                    __m256 b_5 = _mm256_loadu_ps(b+(bj+4)*N+bk);
                    __m256 b_6 = _mm256_loadu_ps(b+(bj+5)*N+bk);
                    __m256 b_7 = _mm256_loadu_ps(b+(bj+6)*N+bk);
                    __m256 b_8 = _mm256_loadu_ps(b+(bj+7)*N+bk);

                    __m256 a_1 = _mm256_set1_ps(a[r*N+bj]);
                    __m256 a_2 = _mm256_set1_ps(a[r*N+bj+1]);
                    __m256 a_3 = _mm256_set1_ps(a[r*N+bj+2]);
                    __m256 a_4 = _mm256_set1_ps(a[r*N+bj+3]);
                    __m256 a_5 = _mm256_set1_ps(a[r*N+bj+4]);
                    __m256 a_6 = _mm256_set1_ps(a[r*N+bj+5]);
                    __m256 a_7 = _mm256_set1_ps(a[r*N+bj+6]);
                    __m256 a_8 = _mm256_set1_ps(a[r*N+bj+7]);

                    b_1 = _mm256_mul_ps(b_1,a_1);
                    b_2 = _mm256_mul_ps(b_2,a_2);
                    b_3 = _mm256_mul_ps(b_3,a_3);
                    b_4 = _mm256_mul_ps(b_4,a_4);
                    b_5 = _mm256_mul_ps(b_5,a_5);
                    b_6 = _mm256_mul_ps(b_6,a_6);
                    b_7 = _mm256_mul_ps(b_7,a_7);
                    b_8 = _mm256_mul_ps(b_8,a_8);

                    __m256 temp_1 = _mm256_add_ps(b_1,b_2);
                    __m256 temp_2 = _mm256_add_ps(b_3,b_4);
                    __m256 temp_3 = _mm256_add_ps(b_5,b_6);
                    __m256 temp_4 = _mm256_add_ps(b_7,b_8);

                    __m256 temp_5 = _mm256_add_ps(temp_1,temp_2);
                    __m256 temp_6 = _mm256_add_ps(temp_3,temp_4);
                    __m256 temp_7 = _mm256_add_ps(temp_5,temp_6);

                    __m256 temp_c = _mm256_loadu_ps(c+r*N+bk);
                    __m256 temp_8 = _mm256_add_ps(temp_7,temp_c);

                    _mm256_storeu_ps(c+r*N+bk,temp_8);
                }
            }
        }
    }
}

void matrix_mul_sse_block(float *a,float *b,float *c,int N){
    int bsize = 32;

#pragma omp parallel for
    for(int i=0;i<N*N;i++)
        c[i]=0.0;
#pragma omp barrier
//    omp_set_nested(1);

//#pragma omp parallel for num_threads(2)
    for(int bi=0; bi<N; bi+=bsize){
#pragma omp parallel for num_threads(4)
        for(int bj=0; bj<N; bj+=bsize){
            for(int bk=0; bk<N; bk++){
                for (int i = bi; i < bi + bsize; i += 4) {
                    for(int j=bj; j<bj + bsize; j+=4) {

                        __m128 b_1 = _mm_loadu_ps(b + i * N + j);
                        __m128 b_2 = _mm_loadu_ps(b + (i + 1) * N + j);
                        __m128 b_3 = _mm_loadu_ps(b + (i + 2) * N + j);
                        __m128 b_4 = _mm_loadu_ps(b + (i + 3) * N + j);

                        __m128 a_1 = _mm_set_ps1(a[bk * N + i]);
                        __m128 a_2 = _mm_set_ps1(a[bk * N + i + 1]);
                        __m128 a_3 = _mm_set_ps1(a[bk * N + i + 2]);
                        __m128 a_4 = _mm_set_ps1(a[bk * N + i + 3]);

                        b_1 = _mm_mul_ps(b_1, a_1);
                        b_2 = _mm_mul_ps(b_2, a_2);
                        b_3 = _mm_mul_ps(b_3, a_3);
                        b_4 = _mm_mul_ps(b_4, a_4);

                        __m128 temp_1 = _mm_add_ps(b_1, b_2);
                        __m128 temp_2 = _mm_add_ps(b_3, b_4);
                        __m128 temp_3 = _mm_add_ps(temp_1, temp_2);

                        __m128 temp_c = _mm_loadu_ps(c + bk * N + j);
                        __m128 temp_4 = _mm_add_ps(temp_3, temp_c);

                        _mm_storeu_ps(c + bk * N + j, temp_4);
                    }
                }
            }
        }
    }
}

void matrix_mul_sse(float *a,float *b,float *c,int N){
    int bsize = 4;

#pragma omp parallel for
    for(int i=0;i<N*N;i++)
        c[i]=0.0;
#pragma omp barrier


    for(int i=0; i<N; i+=bsize){
        for(int j=0; j<N; j+=bsize){
#pragma omp parallel for num_threads(4) schedule(static)
            for(int bk=0; bk<N; bk++){

                        __m128 b_1 = _mm_loadu_ps(b + i * N + j);
                        __m128 b_2 = _mm_loadu_ps(b + (i + 1) * N + j);
                        __m128 b_3 = _mm_loadu_ps(b + (i + 2) * N + j);
                        __m128 b_4 = _mm_loadu_ps(b + (i + 3) * N + j);

                        __m128 a_1 = _mm_set_ps1(a[bk * N + i]);
                        __m128 a_2 = _mm_set_ps1(a[bk * N + i + 1]);
                        __m128 a_3 = _mm_set_ps1(a[bk * N + i + 2]);
                        __m128 a_4 = _mm_set_ps1(a[bk * N + i + 3]);

                        b_1 = _mm_mul_ps(b_1, a_1);
                        b_2 = _mm_mul_ps(b_2, a_2);
                        b_3 = _mm_mul_ps(b_3, a_3);
                        b_4 = _mm_mul_ps(b_4, a_4);

                        __m128 temp_1 = _mm_add_ps(b_1, b_2);
                        __m128 temp_2 = _mm_add_ps(b_3, b_4);
                        __m128 temp_3 = _mm_add_ps(temp_1, temp_2);

                        __m128 temp_c = _mm_loadu_ps(c + bk * N + j);
                        __m128 temp_4 = _mm_add_ps(temp_3, temp_c);

                        _mm_storeu_ps(c + bk * N + j, temp_4);
            }
        }
    }
}


void matrixMultiplyWithSSE(float* a, float * b, float* result, int N){
    __m128 pB, pA,pResult;

#pragma omp parallel for
    for(int i=0;i<N*N;i++)
        result[i]=0.0;
#pragma omp barrier

    int step = 4;
#pragma omp parallel for schedule(static) private(pB,pA,pResult)
    for(int i = 0; i < N; i += step){
        for(int j = 0;j < N; j += step){
            for(int k = 0; k < N; k += step){
                //Above: Partitioning
                //Below: Small matrix multiply
                for(int x = i; x < i + step; x++){
                    pB = _mm_loadu_ps(&b[j * N + k]);
                    pA = _mm_set1_ps(a[x * N + j]);
                    pResult = _mm_mul_ps(pB,pA);
                    pB = _mm_loadu_ps(&b[(j+1) * N + k]);
                    pA = _mm_set1_ps(a[x * N + j+1]);
                    pResult = _mm_add_ps(_mm_mul_ps(pB,pA),pResult);
                    pB = _mm_loadu_ps(&b[(j+2) * N + k]);
                    pA = _mm_set1_ps(a[x * N + j+2]);
                    pResult = _mm_add_ps(_mm_mul_ps(pB,pA),pResult);
                    pB = _mm_loadu_ps(&b[(j+3) * N + k]);
                    pA = _mm_set1_ps(a[x * N + j+3]);
                    pResult = _mm_add_ps(_mm_mul_ps(pB,pA),pResult);
                    _mm_storeu_ps(&result[x * N + k], _mm_add_ps(pResult, _mm_loadu_ps(&result[x * N + k])));
                }
            }
        }
    }
}

void matrix_mul_test(float *a,float *b,float *c,int N){
    int bsize = 8;
//    __m128 pB,pA,pResult;
#pragma omp parallel for
    for(int i=0;i<N*N;i++)
        c[i]=0.0;
#pragma omp barrier

#pragma omp parallel for
    for(int bi=0; bi<N; bi+=bsize){
        for(int bj=0; bj<N; bj+=bsize){
            for(int bk=0; bk<N; bk+=bsize){
                for(int r = bi;r<bi+bsize;r++)
                {
                    __m256 b_1 = _mm256_loadu_ps(b+bj*N+bk);
                    __m256 b_2 = _mm256_loadu_ps(b+(bj+1)*N+bk);
                    __m256 b_3 = _mm256_loadu_ps(b+(bj+2)*N+bk);
                    __m256 b_4 = _mm256_loadu_ps(b+(bj+3)*N+bk);
                    __m256 b_5 = _mm256_loadu_ps(b+(bj+4)*N+bk);
                    __m256 b_6 = _mm256_loadu_ps(b+(bj+5)*N+bk);
                    __m256 b_7 = _mm256_loadu_ps(b+(bj+6)*N+bk);
                    __m256 b_8 = _mm256_loadu_ps(b+(bj+7)*N+bk);

                    __m256 a_1 = _mm256_set1_ps(a[r*N+bj]);
                    __m256 a_2 = _mm256_set1_ps(a[r*N+bj+1]);
                    __m256 a_3 = _mm256_set1_ps(a[r*N+bj+2]);
                    __m256 a_4 = _mm256_set1_ps(a[r*N+bj+3]);
                    __m256 a_5 = _mm256_set1_ps(a[r*N+bj+4]);
                    __m256 a_6 = _mm256_set1_ps(a[r*N+bj+5]);
                    __m256 a_7 = _mm256_set1_ps(a[r*N+bj+6]);
                    __m256 a_8 = _mm256_set1_ps(a[r*N+bj+7]);

                    b_1 = _mm256_mul_ps(b_1,a_1);
                    b_2 = _mm256_mul_ps(b_2,a_2);
                    b_3 = _mm256_mul_ps(b_3,a_3);
                    b_4 = _mm256_mul_ps(b_4,a_4);
                    b_5 = _mm256_mul_ps(b_5,a_5);
                    b_6 = _mm256_mul_ps(b_6,a_6);
                    b_7 = _mm256_mul_ps(b_7,a_7);
                    b_8 = _mm256_mul_ps(b_8,a_8);

                    __m256 temp_1 = _mm256_add_ps(b_1,b_2);
                    __m256 temp_2 = _mm256_add_ps(b_3,b_4);
                    __m256 temp_3 = _mm256_add_ps(b_5,b_6);
                    __m256 temp_4 = _mm256_add_ps(b_7,b_8);

                    __m256 temp_5 = _mm256_add_ps(temp_1,temp_2);
                    __m256 temp_6 = _mm256_add_ps(temp_3,temp_4);
                    __m256 temp_7 = _mm256_add_ps(temp_5,temp_6);

                    __m256 temp_c = _mm256_loadu_ps(c+r*N+bk);
                    __m256 temp_8 = _mm256_add_ps(temp_7,temp_c);

                    _mm256_storeu_ps(c+r*N+bk,temp_8);

//                    pB = _mm_loadu_ps(&b[bj * N + bk]);
//                    pA = _mm_set1_ps(a[r * N + bj]);
//                    pResult = _mm_mul_ps(pB,pA);
//                    pB = _mm_loadu_ps(&b[(bj+1) * N + bk]);
//                    pA = _mm_set1_ps(a[r * N + bj+1]);
//                    pResult = _mm_add_ps(_mm_mul_ps(pB,pA),pResult);
//                    pB = _mm_loadu_ps(&b[(bj+2) * N + bk]);
//                    pA = _mm_set1_ps(a[r * N + bj+2]);
//                    pResult = _mm_add_ps(_mm_mul_ps(pB,pA),pResult);
//                    pB = _mm_loadu_ps(&b[(bj+3) * N + bk]);
//                    pA = _mm_set1_ps(a[r * N + bj+3]);
//                    pResult = _mm_add_ps(_mm_mul_ps(pB,pA),pResult);
//                    _mm_storeu_ps(&c[r * N + bk], _mm_add_ps(pResult, _mm_loadu_ps(&c[r * N + bk])));
                }
            }
        }
    }
}

float compute_trace(float *c, int N){
    float trace = 0.0;
#pragma omp parallel for reduction(+:trace)
    for(int i=0;i<N;i++){
        trace+=c[i*N+i];
    }
    return trace;
}

float get_min(float *c,int N){
    float min = 1<<20;
    for(int i =0; i<N; i++){
        float max =-1;
        for(int j = 0; j<N; j++) {
            if (c[i * N + j] > max)
                max = c[i * N + j];
        }
        if(max<min)
            min = max;
    }
    return min;
}






float get_min_avx(float *c,int N){
    float* results = malloc(sizeof(float)*8);
    float min = 1<<20;
    for(int i =0; i<N; i++){
        float max =-1;
        __m256 result = _mm256_set1_ps(-1024.0);
        for(int j = 0; j<N; j+=8) {
            __m256 temp = _mm256_loadu_ps(&c[i*N+j]);
            result = _mm256_max_ps(result,temp);
        }
        _mm256_storeu_ps(results,result);
        for(int j = 0; j<8; j++){
            if(results[j]>max)
                max = results[j];
        }
        if(max<min){
            min=max;
        }
    }
    return min;
}