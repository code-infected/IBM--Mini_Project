# IBM--Mini_Project

This project is focused on classifying SMS messages as either "ham" (non-spam) or "spam" using a deep learning approach with TensorFlow and Keras.

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Model](#model)
- [Usage](#usage)
- [Visualization](#visualization)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, clone the repository and install the necessary libraries. 

```bash
git clone https://github.com/yourusername/sms-text-classification.git
cd sms-text-classification
pip install -r requirements.txt
```

Make sure you have `wget` installed to download the datasets. You can install it via your package manager. For example, on Ubuntu:

```bash
sudo apt-get install wget
```

## Data

The training and validation datasets can be downloaded using the following commands:

```bash
wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv
```

The data files are in TSV format and consist of two columns: `class` and `message`. The `class` column indicates whether a message is "ham" or "spam".

## Model

The model is a Sequential neural network with the following layers:
- Embedding layer
- Flatten layer
- Dense layer with a sigmoid activation function

It uses binary cross-entropy as the loss function and the Adam optimizer. Early stopping is implemented to prevent overfitting.

## Usage

Run the following script to train the model:

```python
python train_model.py
```

To predict the class of a new message, use the `predict_message` function defined in the script.

## Visualization

The project includes a function to visualize the predictions of the model compared to the actual labels:

```python
def visualize_predictions():
    test_messages = ["how are you doing today",
                     "sale today! to stop texts call 98912460324",
                     "i dont want to go. can we try it a different day? available sat",
                     "our new mobile video service is live. just install on your phone to start watching.",
                     "you have won £1000 cash! call to claim your prize.",
                     "i'll bring it tomorrow. don't forget the milk.",
                     "wow, is your arm alright. that happened to me one time too"
                    ]

    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]

    predictions = [predict_message(msg)[1] for msg in test_messages]

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(test_answers)), test_answers, 'go', label='True Label', markersize=10)
    plt.plot(range(len(predictions)), predictions, 'ro', label='Predicted Label', markersize=10)
    plt.xlabel('Message Index')
    plt.ylabel('Label')
    plt.title('SMS Classification: True vs Predicted Labels')
    plt.xticks(range(len(test_messages)), ['Msg {}'.format(i) for i in range(len(test_messages))], rotation=45)
    plt.legend(loc='upper right')
    plt.show()

visualize_predictions()
```

## Results

The model achieves high accuracy on the validation dataset. Detailed results and model performance metrics are displayed during training.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any feature requests or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
