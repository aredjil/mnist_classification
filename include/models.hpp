#pragma once 
#include <torch/torch.h>
// LeNet architecture 
struct LeNetImpl : torch::nn::Module {
    LeNetImpl() {
        // Convolutional layers
        conv1 = register_module("conv1", torch::nn::Conv2d(1, 6, 5));   // 1 input, 6 output, 5x5 kernel
        conv2 = register_module("conv2", torch::nn::Conv2d(6, 16, 5));  // 6 input, 16 output, 5x5 kernel

        // Fully connected layers
        fc1 = register_module("fc1", torch::nn::Linear(16 * 4 * 4, 120)); // after pooling, image is 4x4
        fc2 = register_module("fc2", torch::nn::Linear(120, 84));
        fc3 = register_module("fc3", torch::nn::Linear(84, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::max_pool2d(x, 2); // pool 2x2
        x = torch::relu(conv2->forward(x));
        x = torch::max_pool2d(x, 2); // pool 2x2
        x = x.view({x.size(0), -1}); // flatten
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), 1);
        return x;
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
TORCH_MODULE(LeNet);
