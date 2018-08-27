/* DESCRIPTION: Trying small example to test whether array is transposed on GPU 
side while 3d CUFFT is being performed dimension wise.
Result of DFT on x-dimension is compared with transposed DFT on host side.
Playing around with cufftPlanMany.
Computes DFT of each ROW of the matrix. Hence it computes N DFTs of size M.
*/


#include<stdio.h>
#include<cufft.h>
#include<cuComplex.h>
#define N 4
#define M 2
#define RANK 1 //only 1d transforms are being computed
int main(){

cuFloatComplex *d_signal, *d_result, *h_signal, *h_result;
int BATCH 4 //the "howmany" parameter in fftw

h_signal= (cuFloatComplex *)malloc(N*M*sizeof(cuFloatComplex));
h_result= (cuFloatComplex *)malloc(N*M*sizeof(cuFloatComplex));

cudaMalloc(&d_signal, N*M*sizeof(cuFloatComplex));
cudaMalloc(&d_result, N*M*sizeof(cuFloatComplex));


for(int i=0; i<N; i++)
 for(int j=0; j<M; j++)
   h_signal[i*M + j] =make_cuFloatComplex( i*M + j,0);

cudaMemcpy(d_signal,h_signal,N*M*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);


//parameters for the "many" transform
int istride, ostride, idist, odist;
cufftHandle plan;


int n[]={2}; //1d transforms of length M
istride =1;
ostride=1;
idist= M;
odist = M;
int *iembed= n, *oembed=n;


cufftPlanMany(&plan, RANK, n, iembed, istride, idist,  oembed, ostride, odist, CUFFT_C2C, BATCH);
    
 cufftExecC2C(plan, d_signal, d_result, CUFFT_FORWARD);
 cudaDeviceSynchronize();
  
 cufftDestroy(plan);
 cudaFree(d_signal);

 cudaMemcpy(h_result, d_result, N*M*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++){
    for (int j = 0; j < M; j++)
      printf("%.3f ", cuCrealf(h_signal[(i*M)+j]));
    printf("\n"); }

  printf("result:\n");


  for (int i = 0; i < N; i++){
    for (int j = 0; j < M; j++)
      printf("%.3f ", cuCrealf(h_result[(i*M)+j]));
    printf("\n"); }

  return 0;



}
