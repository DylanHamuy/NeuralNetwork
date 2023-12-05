// NeuralNetwork.hpp
#include "../include/Eigen/Dense"
#include <iostream>
#include <vector>

#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

// use typedefs for future ease for changing data types like : float to double
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

// neural network implementation class!
class NeuralNetwork {
public:
	// constructor
	NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));

	ColVector sigmoid(const ColVector& x);

	// function for backward propagation of errors made by neurons
	void propagateForward();
	
	void propagateBackward();

	// for given weights/baises, take many training examples, return the average cost
	// avg cost = costs from every example input of a picture for a set weight/bias
	float costs(std::vector<Matrix*> weights, std::vector<ColVector*> biases);

	// a cost for a singular example
	float cost(ColVector& actual, uint expected);

	// function to update the weights of connections
	void updateWeights();

	// function to train the neural network give an array of data points
	void train(std::__1::vector<std::__1::vector<double>> images, std::__1::vector<int> labels);

	uint sample(std::__1::vector<double>);

	// storage objects for working of neural network
	std::vector<ColVector*> neuronLayers; // stores the different layers of out network
	std::vector<Matrix*> weights; // the connection weights itself
    std::vector<ColVector*> biases;
	Scalar learningRate;
    std::vector<uint> topology;
};
#endif // NEURALNETWORK_HPP 

/* NOTES: total weights/bias = 16*784+16+16+10+16*16+16*10 = 13002 */