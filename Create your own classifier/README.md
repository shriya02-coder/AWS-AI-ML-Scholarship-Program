
# Image Classifier Project

This project trains a deep learning model to classify different species of flowers using a dataset of flower images. The trained model can then be used to make predictions on new images. The project includes two main scripts:

1. `train.py`: Trains a new model on a dataset of flower images and saves the trained model as a checkpoint.
2. `predict.py`: Uses the trained model to predict the class of a new image, showing the most likely class and the top K predictions.

## Dependencies

To run the project, you need the following dependencies installed:

- Python 3.6+
- PyTorch
- torchvision
- PIL
- NumPy
- Matplotlib
- JSON

You can install the necessary packages using `pip`:

```bash
pip install torch torchvision pillow numpy matplotlib
```

## Dataset

The dataset used for this project contains images of flowers categorized into 102 different species. The dataset is organized into three parts: `train`, `valid`, and `test`.

After downloading, extract the dataset into a directory named `flowers/`, with the following structure:

```
flowers/
    train/
    valid/
    test/
```

## Training the Model

To train a new model, use the `train.py` script. The script allows you to specify various options such as the model architecture, learning rate, number of epochs, and more.

### Basic Usage:

```bash
python train.py flowers --save_dir . --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 10 --gpu
```

### Arguments:

- `flowers`: Directory containing the dataset (with `train`, `valid`, and `test` subdirectories).
- `--save_dir`: Directory to save the trained model checkpoint (default: current directory).
- `--arch`: Pre-trained model architecture to use (`vgg16` or `densenet121`).
- `--learning_rate`: Learning rate for training the model (default: `0.001`).
- `--hidden_units`: Number of hidden units in the classifier (default: `512`).
- `--epochs`: Number of training epochs (default: `10`).
- `--gpu`: Use GPU for training if available.

### Example:

```bash
python train.py flowers --save_dir checkpoints --arch densenet121 --learning_rate 0.003 --hidden_units 256 --epochs 20 --gpu
```

This command will train the model using the `densenet121` architecture, a learning rate of `0.003`, and 256 hidden units for 20 epochs, saving the model checkpoint in the `checkpoints` directory.

## Predicting the Class of an Image

To predict the class of a new image using the trained model, use the `predict.py` script.

### Basic Usage:

```bash
python predict.py /path/to/image checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
```

### Arguments:

- `/path/to/image`: Path to the input image to be classified.
- `checkpoint.pth`: Path to the model checkpoint file saved during training.
- `--top_k`: Number of top predictions to return (default: `1`).
- `--category_names`: Path to a JSON file mapping category indices to real names.
- `--gpu`: Use GPU for inference if available.

### Example:

```bash
python predict.py flowers/test/1/image_06743.jpg checkpoints/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
```

This command will classify the image `image_06743.jpg`, using the model checkpoint from the `checkpoints` directory, and return the top 5 predictions.

### Output:

The script will print the most likely class and the top K classes with their associated probabilities, for example:

```
Most likely class: hibiscus with probability 0.4217

Top K classes:
1. hibiscus: 0.4217
2. petunia: 0.1648
3. wild pansy: 0.0918
4. pink primrose: 0.0871
5. mallow: 0.0323
```
