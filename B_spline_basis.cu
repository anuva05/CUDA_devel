#include<stdio.h>

__global__  void generate_basis(double x,int i,double *knots,int n){

// x =  1d point
// i =augmented knot index
//knots= 
//n = degree of B spline

int m = n+1;
int N= sizeof(knots);
double alpha1, alpha2;
float *B[N+2*m][n+1];


	if(n==0){

         if(knots[i]<=x) && (x<knots[i+1]){
         B[i][n]=1 ;}
         else{
         B[i][n]=0;
	}
        return(B);
 	}//if n=0

       
       else {
	if((knots[n+i] - knots[i]) == 0) {
		alpha1 = 0;
		} else {
	 	alpha1 = (x - knots[i])/(knots[n+i] - knots[i]);
                 }
	if((knots[i+n+1] - knots[i+1]) == 0) {
	alpha2 = 0;
	} else 
	alpha2 = (knots[i+n+1] - x)/(knots[i+n+1] - knots[i+1]);
}
	B = alpha1*generate_basis(x, (n-1), i, knots) + alpha2*generate_basis(x, (n-1), (i+1), knots);
}
	return(B);



}
