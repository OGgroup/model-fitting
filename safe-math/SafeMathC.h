// Ed Callaghan
// Boring mathematical functions that numerically reflect correct limits
// November 2021

#ifndef __SAFEMATHC_H__
#define __SAFEMATHC_H__

#include <math.h>

double xlogx(double x);
double xlogx_d0(double x);

double expo(double x);

double experfc(double x, double a, double b, double c);
double experfc_d0(double x, double a, double b, double c);
double experfc_d1(double x, double a, double b, double c);
double experfc_d2(double x, double a, double b, double c);
double experfc_d3(double x, double a, double b, double c);

#endif
