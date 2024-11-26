


# Deep Convolutional GAN

[![unlicense](https://img.shields.io/badge/License-Unlicence-6D6490.svg)](https://opensource.org/license/unlicense) [![dcgan](https://img.shields.io/badge/DCGAN-6D6490)](https://paperswithcode.com/method/dcgan)

## 🚀 Demo
After training for over 150 epochs, the model can generate impressive images that resemble human faces. Below are sample outputs from the model:

![dcgan_man](man.png)
![dcgan_women](woman.png)

## ⚙️ Deployment

Pytorch has been chosen as the machine learning framework due to the Tensorflow hellish GPU installation on wsl2

To install the required libraries for the project, run the following command:
```bash
pip install -r requirements.txt
```

After the installation is complete, you can initiate the training process either by opening **train_dcgan.ipynb** in Jupyter Notebook or by executing this command in the terminal:

```bash
python train.py --epoch 100 --batch 128 --tensorboard --cuda --use-pretrained --save-weights-after 1 --save 
```

To see the available arguments and options for customization, use the following command:

```bash
python train.py --help
```

## ✒️ Authors

The project is maintained by [@cxrsedhappy](https://www.github.com/cxrsedhappy)


## 🙏 Acknowledgements
Special thanks to the authors of the original paper on https://paperswithcode.com/method/dcgan for their contributions to the field of generative adversarial networks. Their work serves as the foundational reference for this implementation.
 - [Deep Convolutional GAN](https://paperswithcode.com/method/dcgan)
