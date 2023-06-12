#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "sparse.hpp"
#include "matvecops.hpp"
#include "model.hpp"

void Model::initialize(std::string dataPath, unsigned int numUsers, unsigned int numMovies, unsigned int k, long double lambda) {

    std::cout << "initializing" << std::endl;
    
    // random number generator
    std::default_random_engine generator;
    long double a = pow(2.5 / (long double)k, 0.5);
    std::uniform_real_distribution<long double> distribution(-a, a);

    this->k = k;
    this->lambda = lambda;

    // instantiate vectors for latent factors, weights, counts, dataset
    this->U = new std::vector<long double>[numUsers];
    this->M = new std::vector<long double>[numMovies];
    this->W = new std::vector<long double>[numMovies];
    int* userRatingCounts = new int[numUsers];
    int* movieRatingCounts = new int[numMovies];
    this->R = new SparseVector[numUsers];
    
    // fill vectors
    uint i = 0;
    for (i = 0; i < numUsers; i++) {
        userRatingCounts[i] = 0;
        this->muUsers.push_back(0);
        uint j;
        for (j = 0; j < this->k; j++) {
            this->U[i].push_back(distribution(generator));
        }
    }

    for (i = 0; i < numMovies; i++) {
        movieRatingCounts[i] = 0;
        this->muMovies.push_back(0);
        uint j;
        for (j = 0; j < this->k; j++) {
            this->M[i].push_back(distribution(generator));
        }
        // fill weights
        for (j = 0; j < numMovies; j++) {
            this->W[i].push_back(0);
        }
    }

    std::cout << "Vectors initialized" << std::endl;

    // read data file and calculate means
    std::ifstream f1(dataPath);

    uint movieId, userId;
    long double rating;
    std::string date;
    uint userIdx = 0;
    int numRatings = 0;
    mu = 0;

    // first pass
    std::string line, word;
    if (f1.is_open()) {
        while (std::getline(f1, line)) {
            std::stringstream str(line);
            std::getline(str, word, ',');
            movieId = (uint)std::stoi(word);
            std::getline(str, word, ',');
            userId = (uint)std::stoi(word);
            std::getline(str, word, ',');
            rating = std::stold(word);

            if (this->userIdxs.count(userId) == 0) {
                this->userIdxs[userId] = userIdx;
                userIdx++;
            }

            mu += rating;

            this->muUsers[this->userIdxs[userId]] += rating;
            userRatingCounts[this->userIdxs[userId]] += 1;

            this->R[this->userIdxs[userId]].AddEntry(movieId - 1, rating);

            numRatings++;
        }
        f1.close();
    }

    // normalize means
    mu = mu / (long double) numRatings;
    std::cout << mu << std::endl;

    for (i = 0; i < numUsers; i++) {
        this->muUsers[i] = (this->muUsers[i] / ((long double)userRatingCounts[i])) - mu;
    }
    delete [] userRatingCounts;

    std::cout << "First pass complete" << std::endl;

    // second pass
    std::ifstream f2(dataPath);
    if (f2.is_open()) {
        while (std::getline(f2, line)) {
            std::stringstream str(line);
            std::getline(str, word, ',');
            movieId = (uint)std::stoi(word);
            std::getline(str, word, ',');
            userId = (uint)std::stoi(word);
            std::getline(str, word, ',');
            rating = std::stod(word);


            this->muMovies[movieId - 1] += (rating - mu - this->muUsers[this->userIdxs[userId]]);
            movieRatingCounts[movieId - 1] += 1;
        }
        f2.close();
    }

    // normalize
    for (i = 0; i < numMovies; i++) {
        this->muMovies[i] = this->muMovies[i] / ((long double)movieRatingCounts[i]);
    }
    delete [] movieRatingCounts;

    std::cout << "Second pass complete" << std::endl;
}

void Model::train(std::string trainDataPath, std::string valDataPath, long double lr, long double decay, unsigned int epochs) {
    uint e;
    for (e = 0; e < epochs; e++) {
        // instantiate temp variables, open file
        std::ifstream f(trainDataPath);
        long double loss = 0;
        std::string date;
        std::string line, word;

        if (f.is_open()) {
            int i = 0;
            while (std::getline(f, line)) {
                // temporary variables
                SparseVector bmSave, wSave;
                std::vector<long double> temp1, temp2;
                long double rating, epsilon, wComponent, shrink, bSave;
                uint movieId, userId;

                // read rating data
                std::stringstream str(line);
                std::getline(str, word, ',');
                movieId = (uint)std::stoi(word);
                std::getline(str, word, ',');
                userId = (uint)std::stoi(word);
                std::getline(str, word, ',');
                rating = std::stod(word);

                // calculate neighbors component, save important values
                wComponent = 0;
                bSave = 0;
                unsigned int idx;
                long double r;
                for (unsigned int j = 0; j < R[userIdxs[userId]].Size(); j++) {
                    idx = R[userIdxs[userId]].idxs[j];
                    if (idx != (movieId - 1)) {
                        r = R[userIdxs[userId]].vals[j];
                        wComponent += W[movieId - 1][idx] * (r - mu - muUsers[userIdxs[userId]] - muMovies[idx]);
                        wSave.AddEntry(idx, r - mu - muUsers[userIdxs[userId]] - muMovies[idx]);
                        bmSave.AddEntry(idx, W[movieId - 1][idx]);
                        bSave += W[movieId - 1][idx];
                    }
                }
                shrink = wSave.Size() > 0 ? pow(wSave.Size(), 0.5) : 0;
                wComponent = shrink > 0 ? wComponent / shrink : 0;

                // calculate epsilon
                epsilon = (rating - this->mu - this->muUsers[this->userIdxs[userId]] - this->muMovies[movieId - 1] - wComponent - dot(this->U[this->userIdxs[userId]], this->M[movieId - 1]));

                // update factors
                temp1 = scale(this->U[this->userIdxs[userId]], 2 * lr * epsilon);
                temp2 = scale(this->M[movieId - 1], 2 * lr * lambda);
                temp1 = add(temp2, temp1, true);
                this->M[movieId - 1] = add(this->M[movieId - 1], temp1, true);

                temp1 = scale(this->M[movieId - 1], 2 * lr * epsilon);
                temp2 = scale(this->U[this->userIdxs[userId]], 2 * lr * lambda);
                temp1 = add(temp2, temp1, true);
                this->U[this->userIdxs[userId]] = add(this->U[this->userIdxs[userId]], temp1, true);

                // update weights and biases
                mu = mu - 2 * (-epsilon * ((bSave / shrink) + 1) + (lambda / 3) * mu);
                muUsers[userIdxs[userId]] = muUsers[userIdxs[userId]] - 2 * (-epsilon * ((bSave / shrink) + 1) + (lambda / 3) * muUsers[userIdxs[userId]]);
                for (unsigned int j = 0; j < wSave.Size(); j++) {
                    idx = wSave.idxs[j];
                    muMovies[idx] = muMovies[idx] -2 * lr * (epsilon * bmSave.vals[j] / shrink + (lambda / 3) * muMovies[idx]);
                    W[movieId - 1][idx] = W[movieId - 1][idx] - 2 * lr * (-epsilon * wSave.vals[j] / shrink + lambda * W[movieId - 1][idx]);
                }

                // calculate loss
                loss += pow(epsilon, 2);

                if (i != 0 && i % 1000000 == 0) {
                    std::cout << "epoch " << e;
                    std::cout << " iteration " << i << " running loss " << pow(loss / i, 0.5) << std::endl;
                }

                i++;
            }
            f.close();
        }

        this->predict(valDataPath);
        lr = lr * decay;
    }

}

void Model::predict(std::string dataPath) {
    // instantiate temp variables, open file
    std::ifstream f(dataPath);
    //uint movieId, userId;
    //long double rating, epsilon, wComponent;
    long double loss = 0;
    //std::vector<long double> temp1, temp2;
    std::string date;
    std::string line, word;
    int i = 0;

    if (f.is_open()) {
        while (std::getline(f, line)) {
            // temporary variables
            std::vector<long double> temp1, temp2;
            long double rating, epsilon, wComponent, shrink;
            uint movieId, userId;

            // read rating data
            std::stringstream str(line);
            std::getline(str, word, ',');
            movieId = (uint)std::stoi(word);
            std::getline(str, word, ',');
            userId = (uint)std::stoi(word);
            std::getline(str, word, ',');
            rating = std::stod(word);

            // calculate neighbors component
            wComponent = 0;
            unsigned int idx;
            for (unsigned int j = 0; j < R[userIdxs[userId]].Size(); j++) {
                idx = R[userIdxs[userId]].idxs[j];
                if (idx != (movieId - 1)) {
                    wComponent += W[movieId - 1][idx] * (R[userIdxs[userId]].vals[j] - mu - muUsers[userIdxs[userId]] - muMovies[idx]);
                }
            }
            shrink = R[userIdxs[userId]].Size() > 0 ? pow(R[userIdxs[userId]].Size() - 1, 0.5) : 0;
            wComponent = shrink > 0 ? wComponent / shrink : 0;

            // calculate epsilon
            //epsilon = (rating - this->mu - this->muUsers[this->userIdxs[userId]] - this->muMovies[movieId - 1]);
            epsilon = (rating - this->mu - this->muUsers[this->userIdxs[userId]] - this->muMovies[movieId - 1] - wComponent - dot(this->U[this->userIdxs[userId]], this->M[movieId - 1]));


            // calculate loss
            loss += pow(epsilon, 2);

            i++;
        }
        f.close();
    }
    std::cout << "rmse " << pow(loss / i, 0.5) << std::endl;
}