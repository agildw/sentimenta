# Sentiment Analysis using SVM

This project is a simple implementation of sentiment analysis using Support Vector Machines (SVM). It uses the Pandas library for data handling, scikit-learn for machine learning, Flask for web application development, and Flask-CORS for enabling cross-origin resource sharing.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Download the dataset from [here](https://drive.google.com/file/d/1hA4EZzDsDS_G6FOG3W4JuFZrHUKvqQve/view?usp=share_link) and place it in the same directory as the code, then rename it to `dataset.csv`.

## Usage

To use this project, follow these steps:

- Clone this repository
- Navigate to the directory containing the code
- Run the following command:

```bash
flask run
```

- Open your API client (e.g. Postman) and send a POST request to http://localhost:5000/predict with the following form data:

```json
{
  "review": "This is a good movie"
}
```

Note: The model has already been trained on a dataset, so no additional training is required.
