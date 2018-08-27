#include<stdio.h>
#include<cufft.h>
#include <cuComplex.h>
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
   int *iembed = n, *oembed = n;  
   int rank=1;
   int howmany = 3;
   int idist=1,odist= 1;
   int istride=3,ostride=3;
   cufftComplex** datacpu[10][3], data[10][3];   

   datacpu= (cufftComplex **)malloc(sizeof(cufftComplex)*10*3);
   datacpu[0][1]=6;
   datacpu[1][3]=7;
   
   

    cufftHandle plan;
    
     cudaMalloc((void**)&data, sizeof(cufftComplex)*10*3);
    cudaMemcpy(datacpu,data, 10*3*sizeof(cufftComplex), cudaMemcpyHostToDevice);
    cufftPlanMany(&plan, rank, n, &iembed, istride, idist,  &oembed, ostride, odist, CUFFT_C2C, BATCH);
    


    cufftExecC2C(plan, data, data, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    





    //free memory
    cufftDestroy(plan);
    cudaFree(data);
    printf("fin");
    return 0;
}

