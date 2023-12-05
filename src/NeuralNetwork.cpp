#include "NeuralNetwork.hpp"
#include <iostream>
#include "../include/Eigen/Dense"
#include <fstream>
#include <vector>
#include <cmath>
// use typedefs for future ease for changing data types like : float to double
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

// constructor
NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate) {
    this->learningRate = learningRate;
    this->topology = topology;

    
    for (int i = 0; i < topology.size(); i ++) {
        // initialize neuron layers
        neuronLayers.push_back(new ColVector(topology[i]));
        // initialize baises and weights

        if (i > 0) {
            weights.push_back(new Matrix (topology[i], topology[i-1]));
            weights.back()->setRandom();

            biases.push_back(new ColVector(topology[i]));
            biases.back()->setRandom();
        }
    }
}

ColVector NeuralNetwork::sigmoid(const ColVector& x) {
    return 1.0 / (1.0 + (-x).array().exp());
}

// function for forward propagation of data
void NeuralNetwork::propagateForward() {
    for (int i = 0; i < topology.size() - 1; i++) {
        // Perform matrix-vector multiplication and addition
        (*neuronLayers[i + 1]) = sigmoid(((*weights[i]) * (*neuronLayers[i])) + (*biases[i]));
    }
}

float NeuralNetwork::cost(ColVector& actual, uint expected) {
    float avgCost = 0;
    for (int i = 0; i < actual.size(); i ++) {
        if (i == expected) {
            avgCost += pow(actual[i] - 1, 2);
        } else {
            avgCost += pow(actual[i], 2);
        }
        
    }
    return avgCost;
}


// function for backward propagation of errors made by neurons
void propagateBackward(RowVector& output);

// function to calculate errors made by neurons in each layer
void calcErrors(RowVector& output);

// function to update the weights of connections
void updateWeights();

// function to train the neural network give an array of data points
void train(std::vector<RowVector*> data);

// storage objects for working of neural network
/*
    use pointers when using std::vector<Class> as std::vector<Class> calls destructor of 
    Class as soon as it is pushed back! when we use pointers it can't do that, besides
    it also makes our neural network class less heavy!! It would be nice if you can use
    smart pointers instead of usual ones like this
    */
std::vector<ColVector*> neuronLayers; // stores the different layers of out network
std::vector<RowVector*> cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers
std::vector<RowVector*> deltas; // stores the error contribution of each neurons
std::vector<Matrix*> weights; // the connection weights itself
std::vector<ColVector*> biases;
Scalar learningRate;
std::vector<uint> topology;
