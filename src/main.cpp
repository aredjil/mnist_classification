#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <filesystem>

#include "../include/models.hpp"
#include "../include/nlohmann/json.hpp" // Header only json to store the results and paramters 
                                       // of the exprimenets 
                                        
namespace fs = std::filesystem;

using json = nlohmann::json;

std::string get_timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm* tm_info = std::localtime(&now);

    char buffer[20];
    std::strftime(buffer, 20, "%Y%m%d_%H%M%S", tm_info);
    return std::string(buffer);
}



int main(int argc, char** argv) {
    // Hyperparamters 
    int seed{10}; // Random seed for reproducibility 
    size_t epochs{5}; // Number of training epochs 
    int batch_size{32}; // Batch size
    double lr{0.01}; // Learning rate 

    fs::path results_dir = "../results"; // Path to directory where to save the json results 
    // Check if the directory exists otherwise create it 
    if (!fs::exists(results_dir)) {
    fs::create_directories(results_dir);
    }
    // Time stamped name of the output file
    // It contains both meta data, and the metrics 
    std::string filename = "experiment_" + get_timestamp() + ".json";
    fs::path full_path = results_dir / filename;
    nlohmann::json experiment;
    /*
    FIXME: Pass the name of the expriement from the command line along with hyperparamters  
    */
    experiment["experiment_name"] = "LetNet";
    experiment["hyperparameters"] = {
        {"seed", seed},
        {"batch_size", batch_size},
        {"learning_rate", lr},
        {"optimizer", "SGD"},
        {"epochs", epochs}
    };
    // Fixing the seed
    torch::manual_seed(seed);
    // Selecting GPU device if availbale 
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }
    // Create model and move to device
    LeNet model;
    model.to(device);

    // Load MNIST dataset
    auto train_dataset = torch::data::datasets::MNIST(
            "./data",
            torch::data::datasets::MNIST::Mode::kTrain)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(
            std::move(train_dataset), 
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

    auto test_dataset = torch::data::datasets::MNIST(
            "./data",
            torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    auto test_loader = torch::data::make_data_loader(
            std::move(test_dataset),
            torch::data::DataLoaderOptions().batch_size(1000).workers(2));

    // Optimizer
    torch::optim::SGD optimizer(model.parameters(), lr);

    // ----------------------
    // Training Loop
    // ----------------------
    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        size_t batch_idx = 0;
        float total_loss = 0.0;
        model.train();
        
        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);

            optimizer.zero_grad();
            auto output = model.forward(data);
            auto loss = torch::nll_loss(output, targets);
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
            if (batch_idx++ % 100 == 0) {
                std::cout << "Epoch [" << epoch << "] Batch [" << batch_idx
                          << "] Loss: " << loss.item<float>() << std::endl;
            }
        }
        // ----------------------
        // Evaluation
        // ----------------------
        model.eval();
        torch::NoGradGuard no_grad;
        int64_t correct = 0;
        int64_t total = 0;
        for (auto& batch : *test_loader) {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);

            auto output = model.forward(data);
            auto pred = output.argmax(1);
            correct += pred.eq(targets).sum().item<int64_t>();
            total += targets.size(0);
        }

        std::cout << "Test accuracy: "
                  << static_cast<double>(correct) / total * 100.0
                  << "%" << std::endl;
            std::cout << "Epoch " << epoch << " Average loss: "
                  << total_loss / batch_idx << std::endl;
        float avg_loss = total_loss / batch_idx;
        double test_accuracy = static_cast<double>(correct) / total * 100.0;

        experiment["metrics"]["epoch_" + std::to_string(epoch)] = {
            {"train_loss", avg_loss},
            {"test_accuracy", test_accuracy}
        };        
    }
    std::ofstream file(full_path);
    file << experiment.dump(4); 
    file.close();

}
