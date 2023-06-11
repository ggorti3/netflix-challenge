#ifndef MODEL_HPP
#define MODEL_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include "sparse.hpp"

class Model {

    private:
        unsigned int k;
        long double lambda;
        long double mu;
        std::vector<long double> muUsers;
        std::vector<long double> muMovies;
        std::vector<long double>* U;
        std::vector<long double>* M;

        std::unordered_map<unsigned int, unsigned int> userIdxs;

        SparseVector* R;
    
    public:
        void initialize(std::string dataPath, unsigned int numUsers, unsigned int numMovies, unsigned int k, long double lambda);
        void train(std::string trainDataPath, std::string valDataPath, long double lr, unsigned int epochs);
        void predict(std::string dataPath);

};

#endif