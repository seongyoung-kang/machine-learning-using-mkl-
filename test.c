
#include<stdio.h>
#include"mkl.h"

int main()
{
	double * a;
	double * b;
	double * c;
	
	int i,m,k,n;
	m = 5;
	k = 3;
	n = 2;

	a = (double *)malloc(sizeof(double)*m*k);
	b = (double *)malloc(sizeof(double)*n*k);
	c = (double *)malloc(sizeof(double)*m*n);
	for(i=0;i<n*m;i++)
		c[i] = 2;

	for(i=0;i<n*k;i++)
		b[i] = i;

	for(i=0;i<m*k;i++)
		a[i] = i;

cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,m,n,k, 1, (const double*)a, m, (const double *)b, n, 1.0,c, n);
	
	for(i=0;i<m*n;i++)
	printf("%.2lf \n",c[i]);
	

}
