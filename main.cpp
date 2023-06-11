#include <iostream>
#include <string>

#include "model.hpp"

int main(int argc, char *argv[]) {
    std::string trainDataPath = argv[1];
    std::string valDataPath = argv[2];
    printf("%d\n", sizeof(double)); // some compilers print 8
    printf("%d\n", sizeof(long double)); // some compilers print 16
    Model model;
    model.initialize(trainDataPath, 480189, 17770, 10, 0.1);
    model.train(trainDataPath, valDataPath, 1e-2, 3);
    model.predict(valDataPath);
}