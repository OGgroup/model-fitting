// Ed Callaghan
// Boring mathematical functions that numerically respect correct limits
// November 2021

%module SafeMathC

%begin %{
    #define SWIG_PYTHON_CAST_MODE
%}

// numpy nonsense
%{
    #define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
    import_array();
%}

// typemaps to allow calls with numpy arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* xx, int xxsize)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* rv, int rvsize)};

%{
    #include <SafeMathC.h>
%}
%include <SafeMathC.h>
