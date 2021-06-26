#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mex.h"
#define DEBUG 1

#define datatype double /* type of the elements in y */

/*
Project the vector y onto the closest vector x of same length with 
sum_{n=0}^{N-1}|x[n]|<=a. 
We must have length>=1.
We can have x==y (projection done in place). If x!=y, the arrays x and y must
not overlap, as x is used for temporary calculations before y is accessed.

L. Condat, "Fast Projection onto the Simplex and the
l1 Ball", preprint Hal-01056171, 2014.

*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    size_t length;
    datatype *y, *x, a;

    if (nrhs != 3)
        mexErrMsgTxt("Function needs 3 arguments");

    /* Input */
    y = mxGetPr(prhs[0]);
    x = mxGetPr(prhs[1]);
    a = mxGetScalar(prhs[2]);

    length = mxGetM(prhs[0]);

    if (length != mxGetM(prhs[1]))
        mexErrMsgTxt("y and x must have the same length");
    if (a<=0.0) {
        if (a==0.0) memset(x,0,length*sizeof(datatype));
        return;
    }
    datatype* aux=(x==y ? (datatype*)mxCalloc(length, sizeof(datatype)) : x);
    int auxlength=1; 
    int auxlengthold=-1;    
    double tau=(*aux=(*y>=0.0 ? *y : -*y))-a;
    int i=1;
    for (; i<length; i++) 
        if (y[i]>0.0) {
            if (y[i]>tau) {
                if ((tau+=((aux[auxlength]=y[i])-tau)/(auxlength-auxlengthold))
                <=y[i]-a) {
                    tau=y[i]-a;
                    auxlengthold=auxlength-1;
                }
                auxlength++;
            }
        } else if (y[i]!=0.0) {
            if (-y[i]>tau) {
                if ((tau+=((aux[auxlength]=-y[i])-tau)/(auxlength-auxlengthold))
                <=aux[auxlength]-a) {
                    tau=aux[auxlength]-a;
                    auxlengthold=auxlength-1;
                }
                auxlength++;
            }
        }
    if (tau<=0) {    /* y is in the l1 ball => x=y */
        if (x!=y) memcpy(x,y,length*sizeof(datatype));
        else mxFree(aux);
    } else {
        datatype* aux0=aux;
        if (auxlengthold>=0) {
            auxlength-=++auxlengthold;
            aux+=auxlengthold;
            while (--auxlengthold>=0) 
                if (aux0[auxlengthold]>tau) 
                    tau+=((*(--aux)=aux0[auxlengthold])-tau)/(++auxlength);
        }
        do {
            auxlengthold=auxlength-1;
            for (i=auxlength=0; i<=auxlengthold; i++)
                if (aux[i]>tau) 
                    aux[auxlength++]=aux[i];    
                else 
                    tau+=(tau-aux[i])/(auxlengthold-i+auxlength);
        } while (auxlength<=auxlengthold);
        for (i=0; i<length; i++)
            x[i]=(y[i]-tau>0.0 ? y[i]-tau : (y[i]+tau<0.0 ? y[i]+tau : 0.0)); 
        if (x==y) mxFree(aux0);
    }
    return;
}
