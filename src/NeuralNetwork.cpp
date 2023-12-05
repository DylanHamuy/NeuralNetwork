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
void NeuralNetwork::clearDeltas() {
    for (auto ptr : deltas) {
        delete ptr;
    }
    deltas.clear();
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


ColVector NeuralNetwork::sigmoidDerivative(const ColVector& x) {
    ColVector sigmoidX = sigmoid(x);
    return sigmoidX.array() * (1 - sigmoidX.array());
}

void NeuralNetwork::propagateBackward(uint expected) {
    // Clear previous deltas
    clearDeltas();

    // Calculate the output layer error
    ColVector diff = ColVector::Zero(topology.back());
    diff[expected] = 1;
    ColVector outputError = *neuronLayers.back() - diff;

    // Calculate the output layer delta
    deltas.push_back(new ColVector(outputError.array() * sigmoidDerivative(*neuronLayers.back()).array()));

    // Backpropagate the error to hidden layers
    for (int i = topology.size() - 2; i > 0; --i) {
        // Calculate the error in the current layer
        ColVector error = (weights[i]->transpose() * *deltas.back()).array() * sigmoidDerivative(*neuronLayers[i]).array();
        deltas.push_back(new ColVector(error));
    }

    // Reverse the order of deltas for proper weight update
    std::reverse(deltas.begin(), deltas.end());
}

void NeuralNetwork::updateWeights() {
    for (int i = 0; i < weights.size(); ++i) {
        Matrix weightUpdate = -learningRate * (*deltas[i]) * neuronLayers[i]->transpose();
        weights[i]->noalias() += weightUpdate;
        biases[i]->noalias() += -learningRate * (*deltas[i]);
    }
}

void NeuralNetwork::train(std::vector<std::vector<double>> images, std::vector<int> labels) {
    // Assuming your NeuralNetwork class has a function to set input values
    // Set input values and perform forward and backward propagation for each training example
    for (size_t i = 0; i < images.size(); ++i) {
        setInputs(images[i]);
        propagateForward();
        propagateBackward(labels[i]);
        updateWeights();
        clearDeltas();
    }
}

// Assuming you have a function to set the input values in your NeuralNetwork class
void NeuralNetwork::setInputs(const std::vector<double>& input) {
    for (size_t i = 0; i < input.size(); ++i) {
        (*neuronLayers[0])[i] = input[i];
    }
}

uint NeuralNetwork::sample(const std::vector<double>& input) {
    setInputs(input);
    propagateForward();

    // Assuming you have a function to get the index of the maximum element in a vector
    return std::distance(neuronLayers.back()->data(), std::max_element(neuronLayers.back()->data(), neuronLayers.back()->data() + neuronLayers.back()->size()));
}

// storage objects for working of neural network
/*
    use pointers when using std::vector<Class> as std::vector<Class> calls destructor of 
    Class as soon as it is pushed back! when we use pointers it can't do that, besides
    it also makes our neural network class less heavy!! It would be nice if you can use
    smart pointers instead of usual ones like this
    */
std::vector<ColVector*> neuronLayers; // stores the different layers of out network
std::vector<RowVector*> cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers
std::vector<Matrix*> weights; // the connection weights itself
std::vector<ColVector*> biases;
std::vector<ColVector*> deltas;
Scalar learningRate;
std::vector<uint> topology;
