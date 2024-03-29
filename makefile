CXX := g++

CXXFLAGS := -O3 -std=c++11 -Wall -Wconversion -Wextra -pedantic

main: main.cpp matvecops.cpp sparse.cpp model.cpp
	$(CXX) $(CXXFLAGS) main.cpp matvecops.cpp sparse.cpp model.cpp -o main

.PHONY: clean
clean:
	$(RM) main