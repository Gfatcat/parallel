#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <omp.h>

float rand_float(float s);
void matrix_gen(float *a,float *b,int N,float seed);
float get_min_avx(float *c,int N);
void matrix_mul_avx(float *a,float *b,float *c,int N);

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
    float min;

    //待测试的程序段
    matrix_gen(a,b,N,seed);
    matrix_mul_avx(a,b,c,N);
    min = get_min_avx(c,N);
    printf("%f",min);
    return 0;
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
    for(int j = 0; j<8; j++){
        results[j]=0;
    }
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