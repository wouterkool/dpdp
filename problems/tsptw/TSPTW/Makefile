#CFLAGS := -Wall -g -DDEBUG
LDFLAGS:=
LIBS   := -lpthread
LIBS   := -lm

CXXFLAGS:= -std=c++14
CXXFLAGS:= -O3 -std=c++0x -Wno-write-strings
CXXFLAGS:= -g -std=c++0x -Wno-write-strings # comment out for performance

tsptw_srcs:= \
	gvns.cpp \
	local2opt.cpp \
	localgenius.cpp \
	localsearch.cpp \
	localvnd.cpp \
	mtrandom.cpp \
	tsptwcons.cpp \
	tsptw.cpp \
	tsptwpoint.cpp \
	tsptwsolution.cpp \
	vns.cpp

cpp_srcs:= main.cpp \
	$(tsptw_srcs) \
	tsptwreader.cpp \

cpp_srcs += cputimer.cpp
#cpp_srcs += windowstimer.cpp

tsptw_objs:= $(tsptw_srcs:.cpp=.o)
cpp_objs:= $(cpp_srcs:.cpp=.o)

progs:= Run
libs := libtsptw.a

all: $(cpp_objs) $(progs) $(libs)

Run: $(cpp_objs)

$(progs):
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(libs) : $(tsptw_objs)
	$(AR) cr $@ $^

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CXXFLAGS) -c -o $@ $<

install: $(progs)

clean:
	$(RM) $(progs) *.o *.a *~

dep:
	$(CXX) -MM $(CFLAGS) $(CXXFLAGS) $(cpp_srcs) >> .depend
-include .depend
