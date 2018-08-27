/************************************************
FILENAME: example_paddedpencil.cu

AUTHOR: Anuva K

DESCRIPTION: Test code to perform 3d FFTs on CUDA
according to the proposed pruned framework. FFTs of a small
non-zero subvolume of a larger volume of zeros are to be 
computed pencil by pencil without storing the large 3d array
This script tests X dimension FFT of k x k x k signal to N x k x k
*/ 



#include <cufft.h>
#include <cuComplex.h>
#include <stdio.h>
#define N_SIGS 1 
#define IN_SIG_LEN 8
#define OUT_SIG_LEN 8
int main(){

  cuFloatComplex *h_signal, *d_signal, *h_result, *d_result;

  h_signal = (cuFloatComplex *)malloc(N_SIGS*IN_SIG_LEN*sizeof(cuFloatComplex));
  h_result = (cuFloatComplex *)malloc(N_SIGS*OUT_SIG_LEN*sizeof(cuFloatComplex));
  for (int i = 0; i < N_SIGS; i ++)
    for (int j = 0; j < IN_SIG_LEN/2; j++) // to include padding
  h_signal[(i*IN_SIG_LEN) + j] = make_cuFloatComplex(100*sin((i+1)*6.283*j/IN_SIG_LEN), 0); //this is how to put data into cuFloatComplex type variable
  cudaMalloc(&d_signal, N_SIGS*IN_SIG_LEN*sizeof(cuFloatComplex));
  cudaMalloc(&d_result, N_SIGS*OUT_SIG_LEN*sizeof(cuFloatComplex));

  cudaMemcpy(d_signal, h_signal, N_SIGS*IN_SIG_LEN*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
  cufftHandle plan;
  int n[1] = {IN_SIG_LEN};

  cufftResult res = cufftPlanMany(&plan, 1, n,
     NULL, 1, IN_SIG_LEN,  //advanced data layout, NULL shuts it off. idist=IN_SIG_LEN
     NULL, 1, OUT_SIG_LEN,  //advanced data layout, NULL shuts it off. odist= OUT_SIG_LEN
     CUFFT_C2C, N_SIGS);
  if (res != CUFFT_SUCCESS) {printf("plan create fail\n"); return 1;}

  res = cufftExecC2C(plan, d_signal, d_result, CUFFT_FORWARD);
  if (res != CUFFT_SUCCESS) {printf("forward transform fail\n"); return 1;}
  cudaMemcpy(h_result, d_result, N_SIGS*OUT_SIG_LEN*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N_SIGS; i++){
    for (int j = 0; j < IN_SIG_LEN; j++)
      printf("%.3f ", cuCrealf(h_signal[(i*IN_SIG_LEN)+j]));
    printf("\n"); }

  printf("result:\n");


  for (int i = 0; i < N_SIGS; i++){
    for (int j = 0; j < OUT_SIG_LEN; j++)
      printf("%.3f ", cuCrealf(h_result[(i*OUT_SIG_LEN)+j]));
    printf("\n"); }

  return 0;
}
