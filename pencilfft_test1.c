#include<stdio.h>


#define BATCH 1
int main(){


   // Transform each column of a 2d array with 10 rows and 3 columns:

  //int rank = 1; /* not 2: we are computing 1d transforms */
  //int n[] = {10}; /* 1d transforms of length 10 */
  //int howmany = 3;
  // int idist = odist = 1;
  // int istride = ostride = 3; /* distance between two elements in 
  //
  //
  //                     
  //           the same column */
   int n[]={10};
   int *inembed = n, *oembed = n;  
   int rank=1;
   int howmany = 3;
   int idist=odist= 1;
   int istride=ostride=3;
   
    cufftHandle plan;
    cufftComplex *data[10][3];
    data[1][1]=4;
    data[1][2]=5;
    
    cudaMalloc((void**)&data, sizeof(cufftComplex)*n);
    cufftPlanMany(&plan, rank, n, &iembed, istride, idist, 
        &oembed, ostride, odist, CUFFT_C2C, BATCH);
    


    cufftExecC2C(plan, data, data, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    





    //free memory
    cufftDestroy(plan);
    cudaFree(data);
    printf("fin");
    return 0;
}

