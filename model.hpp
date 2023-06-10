#ifndef MODEL_HPP
#define MODEL_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include "sparse.hpp"

class Model {

    private:
        uint k;
        double lambda;
        double mu;
        std::vector<double> muUsers;
        std::vector<double> muMovies;
        std::vector<double>* U;
        std::vector<double>* M;

        std::unordered_map<uint, uint> userIdxs;

        SparseVector* R;
    
    public:
        void initialize(std::string dataPath, uint numUsers, uint numMovies, uint k, double lambda);
        void train(std::string trainDataPath, std::string valDataPath, double lr, uint epochs);
        void predict(std::string dataPath);

};

#endif