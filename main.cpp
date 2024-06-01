#include <iostream>
#include <string>

#include "model.hpp"

int main(int argc, char *argv[]) {
    std::string trainDataPath = argv[1];
    std::string valDataPath = argv[2];
    std::string allDataPath;
    std::string outDataPath;
    if (argc > 4) {
        allDataPath = argv[3];
        outDataPath = argv[4];
    }
    printf("%d\n", sizeof(double)); // some compilers print 8
    printf("%d\n", sizeof(long double)); // some compilers print 16
    Model model;
    model.initialize(trainDataPath, 480189, 17770, 50, 0.04);
    model.train(trainDataPath, valDataPath, 2e-3, 0.9, 30);
    model.predict(valDataPath);
    if (argc > 4) {
        model.inference(allDataPath, outDataPath);
    }
}