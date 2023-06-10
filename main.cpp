#include <iostream>
#include <string>

#include "model.hpp"

int main(int argc, char *argv[]) {
    std::string trainDataPath = argv[1];
    std::string evalDataPath = argv[2];
    Model model;
    model.initialize(trainDataPath, 480189, 17770, 10, 0.1);
    model.train(trainDataPath, 1e-2, 3);
    model.predict(evalDataPath);
}