#include <iostream>
#include "../include/Eigen/Dense"
#include "NeuralNetwork.hpp"
#include <fstream>
#include <vector>

std::vector<std::vector<double>> readMNISTImages(std::string path) {
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Could not open file: " << path << std::endl;
        exit(EXIT_FAILURE);
    }

    int magic_number, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    std::vector<std::vector<double>> images(num_images, std::vector<double>(rows * cols));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel_value;
            file.read(reinterpret_cast<char*>(&pixel_value), sizeof(pixel_value));
            images[i][j] = pixel_value / 255.0; // Normalize pixel values to the range [0, 1]
        }
    }

    file.close();

    return images;
}

std::vector<int> readMNISTLabels(std::string path) {
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Could not open file: " << path << std::endl;
        exit(EXIT_FAILURE);
    }

    int magic_number, num_labels;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));

    std::vector<int> labels(num_labels);

    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = label;
    }

    file.close();

    return labels;
}

int main() {
    std::string imagesPath = "../MNIST_ORG/train-images.idx3-ubyte";
    std::string labelsPath = "../MNIST_ORG/t10k-labels.idx1-ubyte";

    auto images = readMNISTImages(imagesPath);
    auto labels = readMNISTLabels(labelsPath);

    std::vector<uint> topology = {784, 16, 16, 10};

    // Now you have the MNIST images and labels in 'images' and 'labels' vectors.
    
    return 0;
}
