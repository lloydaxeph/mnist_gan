# Generative Adversarial Network Implementation - MNIST

## 1.0 About
This project is a simple implementation of the paper [Generative Adversarial Net](https://arxiv.org/pdf/1406.2661). 
As stated in the paper, Generative Adversarial Network (GAN) consists of 2 models. The `Generator` model and the `Discriminator` model. 

Using the [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset), the GAN is aimed to generate acurate image representation of digits from **0 - 9**. The following screenshot is an example generated result on a model that was trained with `epoch=50`.

</br>

![epoch_50](https://github.com/lloydaxeph/mnist_gan/assets/158691653/7637d234-6c8f-4ccc-a237-49a5a22ca2be)


## 2.0 Getting Started
### 2.1 Installation
Install the required packages
```
pip3 install -r requirements.txt
```
### 2.2 Training
Training the model is pretty straight forward. Once you have created an instance of `MNISTGAN`, you can immediately use its `train` method to start training.
```python
mnist_gan = MNISTGAN()
mnist_gan.train(epochs=100, batch_size=64, lr=0.001, is_save_images=True)
```

### 2.3 Testing
Similar to testing, you can use the `test` method to generate images using your trained model.

```python
mnist_gan.test(model_path='mymodel.pt')
```
