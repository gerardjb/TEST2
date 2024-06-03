#Compiler and flags declarations
CXX = g++
#Note c++14 was required for some of the static declarations made by pybind11
CXXFLAGS = -O3 -Wall -shared -std=c++14 -fPIC $(shell python3 -m pybind11 --includes)

#Library linking, etc.
LDFLAGS = -lgsl -lgslcblas -larmadillo -ljsoncpp -std=c++14

#For the compiled python binary
TARGET = pgas_bound$(shell python3-config --extension-suffix)

#Directory declarations including for the c++ built executable
BINDIR = bin
OBJDIR = .obj
SRCDIR = src
DIRS = $(BINDIR) $(OBJDIR)

#Soureces and rules for what to make with them
SOURCESXX := $(wildcard $(SRCDIR)/*.cxx)
SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
SOURCES_C := $(filter-out src/bindings.cpp src/Analyzer.cpp, $(SOURCES))
OBJECTS  := $(SOURCES_C:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
EXECUTABLES := $(SOURCESXX:$(SRCDIR)/%.cxx=$(BINDIR)/%)
#SRCS := $(wildcard $(SRCDIR)/*.cpp)

#What we're making and where we're making it
all: $(TARGET) $(DIRS) $(EXECUTABLES)
directories : $(DIRS)

$(DIRS):
	mkdir -p $(DIRS)
	
$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CFLAGS) -c $< -o $@ $(LDFLAGS)
$(EXECUTABLES): $(BINDIR)/% : $(SRCDIR)/%.cxx $(OBJECTS)
	$(CXX) -o $@ $< $(OBJECTS) $(LDFLAGS)
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@ $(LDFLAGS)
	
.PHONY: all clean directories

clean:
	rm -f $(TARGET)
	rm -f $(EXECUTABLES)
	rm -f $(OBJECTS)
