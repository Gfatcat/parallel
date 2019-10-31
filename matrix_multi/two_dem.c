#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <xmmintrin.h>

float rand_float(float s);
void matrix_gen(float **a,float **b,int N,float seed);
void matrix_mul(float **a,float **b,float **c,int N);
float compute_trace(float **c, int N);
void matrix_mul_mp(float **a,float **b,float **c,int N);
void matrix_mul_block(float **a,float **b,float **c,int N);
void matrix_mul_sse(float **a,float **b,float **c,int N);

int main(int argc, char  *argv[]) {
    int N;
    N = atoi(argv[1]);
    float seed;
    seed = atof(argv[2]);
//    int *pstart = (int *)malloc(sizeof(int) *size);
    float **a = malloc(N*sizeof(float) *N);
    float **b = malloc(N*sizeof(float) *N);
    float **c = malloc(N*sizeof(float) *N);
    for (int i=0;i<N;i++){
        a[i]=malloc(sizeof(float) *N);
        b[i]=malloc(sizeof(float) *N);
        c[i]=malloc(sizeof(float) *N);
    };

    struct timeval start;
    struct timeval end;
    float diff,trace;
//    gettimeofday(&start,NULL);
    //待测试的程序段
    printf("%d\n",N);
    printf("%f\n",seed);
    matrix_gen(a,b,N,seed);
//    printf("%f %f\n",a[52],b[34]);
//    matrix_mul_mp(a,b,c,N);

//    printf("\nRunning pure matrix multiply\n");
//    gettimeofday(&start,NULL);
//    matrix_mul(a,b,c,N);
//    trace = compute_trace(c,N);
//    printf("%f\n",trace);
//    gettimeofday(&end,NULL);
//    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
//    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位

    printf("\nRunning matrix multiply with openMP\n");
    gettimeofday(&start,NULL);
    matrix_mul_mp(a,b,c,N);
    trace = compute_trace(c,N);
    printf("%f\n",trace);
    gettimeofday(&end,NULL);
    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位
//
    printf("\nRunning matrix multiply with block\n");
    gettimeofday(&start,NULL);
    matrix_mul_block(a,b,c,N);
    trace = compute_trace(c,N);
    printf("%f\n",trace);
    gettimeofday(&end,NULL);
    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位

//    printf("\nRunning matrix multiply with sse\n");
//    gettimeofday(&start,NULL);
//    matrix_mul_sse(a,b,c,N);
//    trace = compute_trace(c,N);
//    printf("%f\n",trace);
//    gettimeofday(&end,NULL);
//    diff = end.tv_sec-start.tv_sec+ (end.tv_usec-start.tv_usec)/1000000.0;
//    printf("run time is %f\n",diff);//diff 为 function_to_test 的执行时间,以毫秒为单位

    return 0;
}

/*
Input: a, b are the N*N float matrix, 0<seed<1, float
        This function should initialize two matrixs with rand_float()
*/
float rand_float(float s){
    return 4*s*(1-s);
}


void matrix_gen(float **a,float **b,int N,float seed){
    float s=seed;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            s=rand_float(s);
            a[i][j]=s;
            s=rand_float(s);
            b[i][j]=s;
        }
    }
}

void matrix_mul(float **a,float **b,float **c,int N){

//two-dem
    for(int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            c[i][j] = 0.0;
            for (int k=0;k<N;k++){
                c[i][j]+=a[i][k]*b[k][j];
            }
        }
    }
}

void matrix_mul_mp(float **a,float **b,float **c,int N){

//two-dem
#pragma omp parallel for
    for(int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            c[i][j] = 0.0;
            for (int k=0;k<N;k++){
                c[i][j]+=a[i][k]*b[k][j];
            }
        }
    }
}

void matrix_mul_block(float **a,float **b,float **c,int N){
    int bsize = 16;
//    float sum;
#pragma omp parallel for
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            c[i][j]=0.0;
#pragma omp barrier

#pragma omp parallel for
    for(int bi=0; bi<N; bi+=bsize){
        for(int bj=0; bj<N; bj+=bsize){
            for(int bk=0; bk<N; bk++){
                for(int j=bj; j<bj+bsize; j++){
                    float sum=c[bk][j];
                    for (int i=bi; i<bi+bsize; i++){
                        sum+=a[bk][i]*b[i][j];
                    }
                    c[bk][j]=sum;
                }
            }
        }
    }
}

void matrix_mul_sse(float **a,float **b,float **c,int N){
    int bsize = 4;
//    __m128* sum;
#pragma omp parallel for
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            c[i][j]=0.0;
#pragma omp barrier

//#pragma omp parallel for
    for(int bi=0; bi<N; bi+=bsize){
        for(int bj=0; bj<N; bj+=bsize){
            for(int bk=0; bk<N; bk++){
                __m128* b_1 = (__m128*)(*(b+bi)+bj*4);
                __m128* b_2 = (__m128*)(*(b+bi+1)+bj*4);
                __m128* b_3 = (__m128*)(*(b+bi+2)+bj*4);
                __m128* b_4 = (__m128*)(*(b+bi+3)+bj*4);
                __m128 a_1 = _mm_set_ps1(a[bk][bi]);
                __m128 a_2 = _mm_set_ps1(a[bk][bi+1]);
                __m128 a_3 = _mm_set_ps1(a[bk][bi+2]);
                __m128 a_4 = _mm_set_ps1(a[bk][bi+3]);
                * b_1 = _mm_mul_ps(*b_1,a_1);
                * b_2 = _mm_mul_ps(*b_2,a_2);
                * b_3 = _mm_mul_ps(*b_3,a_3);
                * b_4 = _mm_mul_ps(*b_4,a_4);
                __m128 temp_1 = _mm_add_ps(*b_1,*b_2);
                __m128 temp_2 = _mm_add_ps(*b_3,*b_4);
                __m128 temp_3 = _mm_add_ps(temp_1,temp_2);
                __m128 temp_4 = _mm_add_ps(temp_3,*(__m128*)(*(c+bk)+bj*4));
                _mm_storeu_ps(&c[bk][bj],temp_4);
            }
        }
    }
}

float compute_trace(float **c, int N){
    float trace = 0.0;
//#pragma omp parallel for reduction(+:trace)
    for(int i=0;i<N;i++){
        trace+=c[i][i];

    }
    return trace;
}