# Ed Callaghan
# swiggin
# February, October, November 2021

STD=-std=c++11
INCLUDE= -I.
CPPFLAGS=-fPIC -c $(STD) $(INCLUDE)
OBJS=SafeMathC.o
LDFLAGS=

all: backend swig gcc ld

backend: $(OBJS)

swig: SafeMathC.i
	swig -c++ -python SafeMathC.i

gcc: backend swig
	g++ $(CPPFLAGS) $(shell python3-config --includes) SafeMathC_wrap.cxx

ld: gcc
	g++ -shared $(OBJS) SafeMathC_wrap.o -o _SafeMathC.so

debug: debug.cpp $(OBJS)
	g++ -o $@ $(STD) $(INCLUDE) $^

.cpp.o:
	g++ $(CPPFLAGS) $<

clean:
	rm -rf SafeMathC_wrap.cxx SafeMathC.o SafeMathC_wrap.o _SafeMathC.so __pycache__ *.o debug
