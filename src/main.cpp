#include "../include/get_time.hpp"
#include "../include/models.hpp"
#include "../include/nlohmann/json.hpp" // Header only json to store the results and paramters
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
// of the exprimenets

namespace fs = std::filesystem;

using json = nlohmann::json;

int main(int argc, char **argv) {
  // Hyperparamters
  int seed{10};                         // Random seed for reproducibility
  size_t epochs{5};                     // Number of training epochs
  int batch_size{32};                   // Batch size
  double lr{0.01};                      // Learning rate
  std::string experiment_name{"LeNet"}; // Default experiment name
  // Get the hyperparamters from the commandline
  for (int i = 0; i < argc; ++i) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--lr" && i + 1 < argc) {
        lr = std::stof(argv[++i]);
      } else if (arg == "--b" && i + 1 < argc) {
        batch_size = std::stoi(argv[++i]);
      } else if (arg == "--e" && i + 1 < argc) {
        epochs = std::stoi(argv[++i]);
      } else if (arg == "--name" && i + 1 < argc) {
        experiment_name = argv[++i];
      }
    }
  }

  fs::path results_dir =
      "../results"; // Path to directory where to save the json results
  // Check if the directory exists otherwise create it
  if (!fs::exists(results_dir)) {
    fs::create_directories(results_dir);
  }
  fs::path weights_dir =
      "../models"; // Path to directory where to save the model weights
  // Check if the directory exists otherwise create it
  if (!fs::exists(weights_dir)) {
    fs::create_directories(weights_dir);
  }
  // Time stamped name of the output file
  // It contains both meta data, and the metrics
  std::ostringstream fname;
  fname << experiment_name << "_lr" << lr << "_b" << batch_size << "_e"
        << epochs
        << "_"
        // << get_timestamp()
        << ".json";

  fs::path full_path = results_dir / fname.str();

  nlohmann::json experiment;

  experiment["experiment_name"] = "LetNet";
  experiment["date"] = get_formatted_datetime();
  experiment["hyperparameters"] = {{"seed", seed},
                                   {"batch_size", batch_size},
                                   {"learning_rate", lr},
                                   {"optimizer", "SGD"},
                                   {"epochs", epochs}};
  // Fixing the seed
  torch::manual_seed(seed);
  // Selecting GPU device if availbale
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  }
  // Load MNIST dataset
  auto train_dataset =
      torch::data::datasets::MNIST("./data",
                                   torch::data::datasets::MNIST::Mode::kTrain)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());

  auto train_loader = torch::data::make_data_loader(
      std::move(train_dataset),
      torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

  auto test_dataset =
      torch::data::datasets::MNIST("./data",
                                   torch::data::datasets::MNIST::Mode::kTest)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());

  auto test_loader = torch::data::make_data_loader(
      std::move(test_dataset),
      torch::data::DataLoaderOptions().batch_size(1000).workers(2));

  // Create model and move to device
  LeNet model;
  model->to(device);

  // Optimizer
  torch::optim::SGD optimizer(model->parameters(), lr);


  auto start = std::chrono::high_resolution_clock::now();
  for (size_t epoch = 1; epoch <= epochs; ++epoch) {
    size_t batch_idx = 0;
    float total_loss = 0.0;
    model->train();

    for (auto &batch : *train_loader) {
      auto data = batch.data.to(device);
      auto targets = batch.target.to(device);

      optimizer.zero_grad();
      auto output = model->forward(data);
      auto loss = torch::nll_loss(output, targets);
      loss.backward();
      optimizer.step();

      total_loss += loss.item<float>();
      if (batch_idx++ % 100 == 0) {
        std::cout << "Epoch [" << epoch << "] Batch [" << batch_idx
                  << "] Loss: " << loss.item<float>() << std::endl;
      }
    }

    model->eval();
    torch::NoGradGuard no_grad;
    int64_t correct = 0;
    int64_t total = 0;
    for (auto &batch : *test_loader) {
      auto data = batch.data.to(device);
      auto targets = batch.target.to(device);

      auto output = model->forward(data);
      auto pred = output.argmax(1);
      correct += pred.eq(targets).sum().item<int64_t>();
      total += targets.size(0);
    }

    std::cout << "Test accuracy: "
              << static_cast<double>(correct) / total * 100.0 << "%"
              << std::endl;
    std::cout << "Epoch " << epoch
              << " Average loss: " << total_loss / batch_idx << std::endl;
    float avg_loss = total_loss / batch_idx;
    double test_accuracy = static_cast<double>(correct) / total * 100.0;

    experiment["metrics"]["epoch_" + std::to_string(epoch)] = {
        {"train_loss", avg_loss}, {"test_accuracy", test_accuracy}};
  }
  auto end = std::chrono::high_resolution_clock::now();
  
  std::chrono::duration<double> elapsed = end - start;
  experiment["duration"] = elapsed.count();
  std::ostringstream model_fname;
  model_fname << experiment_name << "_lr" << lr << "_b" << batch_size << "_e"
              << epochs
              << "_"
              //   << get_timestamp()
              << ".pt";

  fs::path model_full_path = weights_dir / model_fname.str();
  // torch::save(model, model_full_path);
  std::ofstream file(full_path);
  file << experiment.dump(4);
  file.close();
  std::cout << "Training took " << elapsed.count() << " seconds.\n";
  std::cout << "Model saved to: " << model_full_path << std::endl;
  std::cout << "Results saved to: " << full_path << std::endl;
}
