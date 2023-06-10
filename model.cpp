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

void Model::initialize(std::string dataPath, uint numUsers, uint numMovies, uint k, double lambda) {

    std::cout << "initializing" << std::endl;
    
    // random number generator
    std::default_random_engine generator;
    double a = pow(2.5 / double(k), 0.5);
    std::uniform_real_distribution<double> distribution(-a, a);

    this->k = k;
    this->lambda = lambda;

    // instantiate vectors for latent factors and counts
    this->U = new std::vector<double>[numUsers];
    this->M = new std::vector<double>[numMovies];
    int* userRatingCounts = new int[numUsers];
    int* movieRatingCounts = new int[numMovies];
    
    // fill vectors
    double muUserFillValue = 1 / double(numUsers);
    uint i = 0;
    for (i = 0; i < numUsers; i++) {
        userRatingCounts[i] = 0;
        this->muUsers.push_back(muUserFillValue);
        uint j;
        for (j = 0; j < this->k; j++) {
            this->U[i].push_back(distribution(generator));
        }
    }

    double muMovieFillValue = 1 / double(numMovies);
    for (i = 0; i < numMovies; i++) {
        movieRatingCounts[i] = 0;
        this->muMovies.push_back(muMovieFillValue);
        uint j;
        for (j = 0; j < this->k; j++) {
            this->M[i].push_back(distribution(generator));
        }
    }

    std::cout << "Vectors initialized" << std::endl;

    // read data file and calculate means
    std::ifstream f(dataPath);

    uint movieId, userId;
    double rating;
    std::string date;
    uint userIdx = 0;
    int numRatings = 0;

    std::string line, word;
    if (f.is_open()) {
        while (std::getline(f, line)) {
            std::stringstream str(line);
            std::getline(str, word, ',');
            movieId = (uint)std::stoi(word);
            std::getline(str, word, ',');
            userId = (uint)std::stoi(word);
            std::getline(str, word, ',');
            rating = std::stod(word);

            if (this->userIdxs.count(userId) == 0) {
                this->userIdxs[userId] = userIdx;
                userIdx++;
            }

            mu += rating;

            this->muMovies[movieId - 1] += rating;
            movieRatingCounts[movieId - 1] += 1;

            this->muUsers[this->userIdxs[userId]] += rating;
            userRatingCounts[this->userIdxs[userId]] += 1;

            if (numRatings % 1000000 == 0) {
                std::cout << "iteration " << numRatings << std::endl;
            }

            numRatings++;
        }
        f.close();
    } else {
        std::cout << "o no" << std::endl;
    }

    // normalize means
    mu = mu / numRatings;

    for (i = 0; i < numMovies; i++) {
        this->muMovies[i] = this->muMovies[i] / double(movieRatingCounts[i]);
    }

    for (i = 0; i < numUsers; i++) {
        this->muUsers[i] = this->muUsers[i] / double(userRatingCounts[i]);
    }

    std::cout << mu << std::endl;

    // free temporary arrays
    delete [] userRatingCounts;
    delete [] movieRatingCounts;
}

void Model::train(std::string dataPath, double lr, uint epochs) {
    uint e;
    for (e = 0; e < epochs; e++) {
        // instantiate temp variables, open file
        std::ifstream f(dataPath);
        uint movieId, userId;
        double rating, epsilon;
        double loss = 0;
        std::vector<double> temp1, temp2;
        std::string date;
        std::string line, word;

        if (f.is_open()) {
            int i = 0;
            while (std::getline(f, line)) {
                // read rating data
                std::stringstream str(line);
                std::getline(str, word, ',');
                movieId = (uint)std::stoi(word);
                std::getline(str, word, ',');
                userId = (uint)std::stoi(word);
                std::getline(str, word, ',');
                rating = std::stod(word);

                // calculate epsilon
                epsilon = (rating - this->mu - dot(this->U[this->userIdxs[userId]], this->M[movieId - 1]));

                // update factors
                temp1 = scale(this->U[this->userIdxs[userId]], 2 * lr * epsilon);
                temp2 = scale(this->M[movieId - 1], 2 * lr * lambda);
                temp1 = add(temp2, temp1, true);
                this->M[movieId - 1] = add(this->M[movieId - 1], temp1, true);

                temp1 = scale(this->M[movieId - 1], 2 * lr * epsilon);
                temp2 = scale(this->U[this->userIdxs[userId]], 2 * lr * lambda);
                temp1 = add(temp2, temp1, true);
                this->U[this->userIdxs[userId]] = add(this->U[this->userIdxs[userId]], temp1, true);

                // calculate loss
                loss += pow(epsilon, 2);

                if (i != 0 && i % 1000000 == 0) {
                    std::cout << "iteration " << i << " running loss " << loss / i << std::endl;
                }

                i++;
            }
            f.close();
        }
    }

}

void Model::predict(std::string dataPath) {
    // instantiate temp variables, open file
    std::ifstream f(dataPath);
    uint movieId, userId;
    double rating, epsilon;
    double loss = 0;
    std::vector<double> temp1, temp2;
    std::string date;
    std::string line, word;
    int i = 0;

    if (f.is_open()) {
        while (std::getline(f, line)) {
            // read rating data
            std::stringstream str(line);
            std::getline(str, word, ',');
            movieId = (uint)std::stoi(word);
            std::getline(str, word, ',');
            userId = (uint)std::stoi(word);
            std::getline(str, word, ',');
            rating = std::stod(word);

            // calculate epsilon
            epsilon = (rating - this->mu - dot(this->U[this->userIdxs[userId]], this->M[movieId - 1]));

            // calculate loss
            loss += pow(epsilon, 2);

            i++;
        }
        f.close();
    }
    std::cout << "rmse " << pow(loss / i, 0.5) << std::endl;
}