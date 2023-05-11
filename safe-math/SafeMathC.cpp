// Ed Callaghan
// Boring mathematical functions that numerically reflect correct limits
// November 2021

#include <SafeMathC.h>
const double PI = 3.141592654;

// product which respects the limit f -> 0 as x -> 0
double xlogx(double x){
    double rv = x;
    if (0.0 < rv){
        rv *= log(x);
    }
    return rv;
}

// derivative of the above
double xlogx_d0(double x){
    double rv = 1.0 + log(x);
    return rv;
}

// truncated exponential --- always <= max float
double expo(double x){
    if (88 < x){
        x = 88;
    }
    double rv = exp(x);
    return rv;
}

// product which respects the limit f -> 0 as x -> -infinity
double experfc(double x, double a, double b, double c){
    double rv = erfc(c - x/b);
    if (0.0 < rv){
        rv *= exp(-x/a);
    }
    return rv;
}

// derivatives of the above
double experfc_d0(double x, double a, double b, double c){
    double term = c - x/b;
    double rv = 1.0;
    rv *= -2*term;
    rv *= -1.0/b;
    rv *= exp(-x/a);
    rv *= exp(-pow(term, 2));
    rv *= 2.0/sqrt(PI);
    return rv;
}

double experfc_d1(double x, double a, double b, double c){
    double rv = (x/a/a) * experfc(x, a, b, c);
    return rv;
}

double experfc_d2(double x, double a, double b, double c){
    double term = c - x/b;
    double rv = 1.0;
    rv *= -2*term;
    rv *= (x/b/b);
    rv *= exp(-x/a);
    rv *= exp(-pow(term, 2));
    rv *= 2.0/sqrt(PI);
    return rv;
}

double experfc_d3(double x, double a, double b, double c){
    double term = c - x/b;
    double rv = 1.0;
    rv *= -2*term;
    rv *= 1.0;
    rv *= exp(-x/a);
    rv *= exp(-pow(term, 2));
    rv *= 2.0/sqrt(PI);
    return rv;
}
