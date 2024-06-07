Here’s an improved version of your README.md file:

---

# Mushroom Edibility Classification

Welcome to the Mushroom Edibility Classification project! This project aims to determine whether a mushroom is safe to eat based on various characteristics.

## Project Overview

The classification is based on the following features:
- Cap Diameter
- Cap Shape
- Gill Attachment
- Gill Color
- Stem Height
- Stem Width
- Stem Color
- Season
- Class (Edible or Poisonous)

## Prerequisites

### Install Python

First, ensure you have Python installed. Run the following commands in your terminal:

```bash
sudo apt update
sudo apt install python3
```

### Install Required Libraries

Navigate to the project directory and install the necessary Python libraries using:

```bash
pip install -r requirements.txt
```

## Running the Project

With all dependencies installed, you can run the project by navigating to the project directory and executing:

```bash
python3 main.py
```

## Viewing the Results

Once the code has finished executing, you can view the results, check the models, and their parameters using MLflow. Start the MLflow UI with:

```bash
mlflow ui
```

Open a web browser and go to `http://localhost:5000` to access the MLflow UI.

## Alternative Method

You can also run the `client.py` program to see the models directly:

```bash
python3 client.py
```

## Project Structure

Here’s a brief overview of the project structure:

```
mushroom-edibility-classification/
│
├── data/
│   ├── mushrooms.csv                # Dataset
│
├──main.py                      # Main script to run the project
│
├── src/
│   ├── client.py                    # Alternative client script to view models
│   ├── model_dealing.py            # Script for training models
│   ├── dataset_treatment.py         # Script for feature selection
│
├── requirements.txt                 # Required Python libraries
├── README.md                        # Project README
├── mlruns/                          # MLflow tracking directory
│
└── 
```

## Contributing

I welcome contributions! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.
