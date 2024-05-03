# DCGAN Training for Image Generation of Traffic Sign

## Introduction
Deep Convolutional Generative Adversarial Networks (DCGANs) represent a significant advancement in the field of generative models, offering a framework for training adversarial networks to generate high-quality images. This project proposes to explore the application of an enhanced DCGAN model with spectral normalization (SN) to the German Traffic Sign Recognition Benchmark (GTSRB), a dataset not previously utilized in foundational DCGAN studies. 

Two DCGAN models will be trained: a baseline model adhering to the original paper, and an enhanced model incorporating SN in the discriminator. The model will be evaluated based on the quality of generated images, the stability of the training process, and their performance in terms of convergence speed. The exploration of the latent space will also aim to understand the diversity and realism of the generated traffic signs.

## Dependencies
- Python 3.x
- PyTorch
- torchvision
- piq
- wandb
- NumPy
- matplotlib (optional, for visualization)

## Installation
1. **Clone the repository:**
   ```bash
   git clone [repository-url]
   cd [repository-directory]
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Configuration:**
   Before running the training script, make sure to configure the `WANDB_API_KEY` in your environment variables or directly in the script to enable logging and monitoring through Weights & Biases.

2. **Training the model:**
   Run the following command to start the training process:
   ```bash
   python train.py
   ```

3. **Evaluating the model:**
   Use the checkpoints saved during training to load the model and evaluate its performance using the provided evaluation scripts.

4. **Generating Images:**
   After training, use the interpolation script to generate and save a series of interpolated images between two points in the latent space.

## Code Structure
- `Generator`: Defines the generator architecture for the DCGAN.
- `Discriminator`: Defines the discriminator architecture for the DCGAN.
- `Discriminator_SN`: Defines the discriminator architecture for the DCGAN with spectral normalization.
- `Trainer`: Manages the training process, including both generator and discriminator updates, logging, and evaluation.
- `save_checkpoint` and `load_checkpoint`: Functions to save and load model state.
- `interpolate_and_generate`: Generates interpolated images between two random points in the latent space.

## Visualization
After training, you can visualize the generated images using the following Python snippet:
```python
from IPython.display import Image
Image(filename='path_to_generated_image.png')
```