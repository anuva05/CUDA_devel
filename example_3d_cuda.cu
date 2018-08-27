/************************************************
FILENAME: example3dcuda.cu

AUTHOR: Anuva K

DESCRIPTION: Transfer subvolume to GPU. Allocate memory for pencil signal.
Copy part of signal into the allocated signal. The rest is zero padding. Compute pencil FFT.
for now, let us consider 1d signal
*/


#include <cufft.h>
#include <cuComplex.h>
#include <stdio.h>
#define N_SIGS 16
#define IN_SIG_LEN 4
#define OUT_SIG_LEN 8
int main(){

  cuFloatComplex *h_signal, *h_pencil, *d_signal, *d_pencil, *h_result, *d_result;
  
  h_signal = (cuFloatComplex *)malloc(N_SIGS*IN_SIG_LEN*sizeof(cuFloatComplex));
  h_result = (cuFloatComplex *)malloc(N_SIGS*OUT_SIG_LEN*sizeof(cuFloatComplex));
  h_pencil = (cuFloatComplex *)malloc(N_SIGS*OUT_SIG_LEN*sizeof(cuFloatComplex));


  for (int i = 0; i < N_SIGS; i ++)
    for (int j = 0; j < IN_SIG_LEN; j++) // to include padding
     h_signal[(i*IN_SIG_LEN) + j] = make_cuFloatComplex(100*sin((i+1)*6.283*j/IN_SIG_LEN), 0); //this is how to put data into cuFloatComplex type variable


  cudaMalloc(&d_signal, N_SIGS*IN_SIG_LEN*sizeof(cuFloatComplex));//same size as input subvolume
  cudaMalloc(&d_pencil, N_SIGS*OUT_SIG_LEN*sizeof(cuFloatComplex));//allocate for pencil
  cudaMalloc(&d_result, N_SIGS*OUT_SIG_LEN*sizeof(cuFloatComplex)); //size of full length pencil
  
  cudaMemcpy(d_signal, h_signal, N_SIGS*IN_SIG_LEN*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

   for(int i=0; i< N_SIGS; i++)
    for(int j=0; j< OUT_SIG_LEN; j++)   
         h_pencil[(i*OUT_SIG_LEN)+j]=make_cuFloatComplex(0,0);
  
   cudaMemcpy(d_pencil, h_pencil, N_SIGS*OUT_SIG_LEN*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

// COPY subvolume pencil into full pencil, for each
   for(int i=0; i< N_SIGS; i++)
    for(int j=0; j< IN_SIG_LEN; j++)  
//       d_pencil[(i*OUT_SIG_LEN)+j]= d_signal[(i*IN_SIG_LEN) + j];
       cudaMemcpy(&d_pencil[(i*OUT_SIG_LEN)+j],&d_signal[(i*IN_SIG_LEN) + j], sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);
  cufftHandle plan;
  

  int n[1] = {OUT_SIG_LEN};

  cufftResult res = cufftPlanMany(&plan, 1, n,
     NULL, 1, OUT_SIG_LEN,  //advanced data layout, NULL shuts it off. idist=IN_SIG_LEN
     NULL, 1, OUT_SIG_LEN,  //advanced data layout, NULL shuts it off. odist= OUT_SIG_LEN
     CUFFT_C2C, N_SIGS);
  if (res != CUFFT_SUCCESS) {printf("plan create fail\n"); return 1;}

  res = cufftExecC2C(plan, d_pencil, d_result, CUFFT_FORWARD);



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
