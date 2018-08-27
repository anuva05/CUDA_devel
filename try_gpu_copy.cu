#include<stdio.h>

int main(){

 int *a =[1,2,3,4];

 int *dev_a;

 cudaMalloc(&dev_a, 4*sizeof(int));

 cudaMemcpy(dev_a, a, 4*sizeof(int), cudaMemcpyHostToDevice);

 prin

 return 0;
}
