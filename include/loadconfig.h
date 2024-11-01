#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

cv::Mat parseMatrixFromYAML(const std::string& filepath, const std::string& key) {
    std::fstream file(filepath, std::ios::in);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open the file!" << std::endl;
        return cv::Mat();
    }

    std::string line;
    cv::Mat matrix;
    bool keyFound = false;
    int rows = 0, cols = 0;
    std::string dt;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string currentKey, value;
        std::getline(ss, currentKey, ':');
        std::getline(ss, value);

        // Trim any leading/trailing whitespace
        currentKey.erase(0, currentKey.find_first_not_of(" \t\n\r"));
        currentKey.erase(currentKey.find_last_not_of(" \t\n\r") + 1);

        if (currentKey == key) {
            keyFound = true;
        }
        else if (keyFound) {
            // Parse the rows, cols, and data type (dt) based on the following lines
            if (currentKey == "rows") {
                rows = std::stoi(value);
            }
            else if (currentKey == "cols") {
                cols = std::stoi(value);
            }
            else if (currentKey == "dt") {
                dt = value;
            }
            else if (currentKey == "data") {
                std::vector<double> matData;
                std::string dataString = line.substr(line.find("[") + 1);
                dataString.pop_back();  // Remove the trailing ']'

                std::stringstream dataStream(dataString);
                double val;
                while (dataStream >> val) {
                    matData.push_back(val);
                    if (dataStream.peek() == ',')
                        dataStream.ignore();
                }

                // Create matrix according to the parsed data
                matrix = cv::Mat(rows, cols, CV_64F, matData.data()).clone();
                break; // Exit after parsing the matrix
            }
        }
    }

    file.close();
    if (!keyFound) {
        std::cerr << "Key not found in the file!" << std::endl;
    }
    return matrix;
}

double parseDoubleFromYAML(const std::string& filepath, const std::string& key) {
    std::fstream file(filepath, std::ios::in);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file!" << std::endl;
        return 0.0;
    }

    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string currentKey, value;
        std::getline(ss, currentKey, ':');
        std::getline(ss, value);

        if (currentKey.find(key) != std::string::npos) {
            return std::stod(value);  // Convert the value string to double and return
        }
    }

    file.close();
    return 0.0;  // Return default value if key is not found
}


int parseIntFromYAML(const std::string& filepath, const std::string& key) {
    std::fstream file(filepath, std::ios::in);
    //cout<<filepath<<endl;
    if (!file.is_open()) {
        std::cerr << "Failed to open the file!" << std::endl;
        return 0;
    }

    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string currentKey, value;
        std::getline(ss, currentKey, ':');
        std::getline(ss, value);

        if (currentKey.find(key) != std::string::npos) {
            return std::stoi(value);  // Convert the value string to int and return
        }
    }

    file.close();
    return 0;  // Return default value if key is not found
}


std::string parseStringFromYAML(const std::string& filepath, const std::string& key) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file!" << std::endl;
        return "";
    }

    std::string line;
    while (std::getline(file, line)) {
        // Remove leading whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r"));

        // Check if the line contains the key followed by a colon
        if (line.find(key + ":") == 0) {
            // Find the position of the colon
            size_t colonPos = line.find(':');
            if (colonPos != std::string::npos) {
                // Extract the value after the colon
                std::string value = line.substr(colonPos + 1);

                // Trim leading and trailing whitespace from the value
                value.erase(0, value.find_first_not_of(" \t\n\r"));
                value.erase(value.find_last_not_of(" \t\n\r") + 1);

                return value;
            }
        }
    }

    file.close();
    return "";  // Return empty string if key is not found
}
