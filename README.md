# Human-Centered-xAI
Introduce explainability into deep learning models

## Requirements
The library versions used here are:
- python=3.8
- pytorch=1.11.0
- cudatoolkit=11.3
- fastai=2.7.9
- fastcore=1.5.24
- ipykernel=6.25.0
- matplotlib=3.5.2
- numpy=1.22.3
- scikit-learn=1.1.0
- seaborn=0.11.2
- torchmetrics=0.7.3

All the required packages could be directly installed within your conda environment by using the file environment.yml through:
```
conda env create -n <your environment name> -f environment.yml
```

## Tutorial
### 1. Define your data
We provide an example of cat/dog classification from *Oxford-IIIT Pet Dataset*.
```
data_path = untar_data(URLs.PETS)
print(data_path.ls())

catblock = MultiCategoryBlock(encoded=True, vocab=['cat', 'dog'])
dblock = DataBlock(
    blocks=(ImageBlock(), catblock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=label_func,
    item_tfms=Resize(128),
    batch_tfms=[Normalize.from_stats(*imagenet_stats)],
)
```
With label_func defined as:
```
def label_func(f): 
    name = f.name #on veut accéder aux noms uniquement
    if name[0].isupper(): #on veut tester la première lettre uniquement donc on applique "isupper" au premier élément de name (name[0])
        lab = torch.tensor([1, 0], dtype=torch.float32)
    else:
        lab = torch.tensor([0, 1], dtype=torch.float32)
    return lab
```
The data loader is therefore created using:
```
dls = dblock.dataloaders(data_path/"images", bs=16, drop_last=True)
```

### 2. Define your model
The general architecture is an Adversarial AutoEncoder (AAE), as shown in the following figure:
![Architecture](https://github.com/LucaLaFisca/Human-Centered-xAI/blob/main/images/architecture.png)
The model MUST be an autoencoder and additional fully connected layers should be defined for the classifier and the discriminator.
You can modify the provided one in the [model.py](https://github.com/LucaLaFisca/Human-Centered-xAI/blob/main/model.py) file.
```
def __init__(
    self,
    input_size,
    input_channels,
    encoding_dims=128,
    step_channels=16,
    nonlinearity=nn.LeakyReLU(0.2),
    classes=2,
    gen_train=True
):
    super(AAEGen, self).__init__()

    self.gen_train = gen_train
    self.count_acc = 1
    self.classes = classes

    self.pool = nn.AdaptiveAvgPool2d(1)
    self.flatten = nn.Flatten()
    self.dropout = nn.Dropout(p=0.2)#, inplace=True)
    # self.linear = nn.Linear(self.encoder.out_channels[-1], 2, bias=True) #2 classes
    self.linear = nn.Linear(encoding_dims, self.classes, bias=True) #8 classes
    self.bn_lin = nn.BatchNorm1d(num_features=encoding_dims)

    self.fc_crit1 = nn.Linear(encoding_dims*2, 64)
    self.fc_crit2 = nn.Linear(64, 16)
    self.fc_crit3 = nn.Linear(16, 1)

    self.bn_crit1 = nn.BatchNorm1d(num_features=64)
    self.bn_crit2 = nn.BatchNorm1d(num_features=16)

    encoder = [
        nn.Sequential(
            nn.Conv2d(input_channels, step_channels, 5, 2, 2), nonlinearity
        )
    ]
    size = input_size // 2
    channels = step_channels
    while size > 1:
        encoder.append(
            nn.Sequential(
                nn.Conv2d(channels, channels * 4, 5, 4, 2),
                nn.BatchNorm2d(channels * 4),
                nonlinearity,
            )
        )
        channels *= 4
        size = size // 4
    self.encoder = nn.Sequential(*encoder)
    self.encoder_fc = nn.Linear(
        channels, encoding_dims
    )  # Can add a Tanh nonlinearity if training is unstable as noise prior is Gaussian
    self.decoder_fc = nn.Linear(encoding_dims, step_channels)
    decoder = []
    size = 1
    channels = step_channels
    while size < input_size // 2:
        decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(channels, channels * 4, 5, 4, 2, 3),
                nn.BatchNorm2d(channels * 4),
                nonlinearity,
            )
        )
        channels *= 4
        size *= 4
    decoder.append(nn.ConvTranspose2d(channels, input_channels, 5, 2, 2, 1))
    self.decoder = nn.Sequential(*decoder)
```

Then, in the main.py file, you can instantiate it using:
```
model = AAE(
        input_size=128,
        input_channels=3,
        encoding_dims=128,
        classes=2,
)
```

### 3. Train the model and display the resulting latent space
Run the main.py script in your command prompt
```
python main.py
```

## Results
The result provided by this repository consists of the final latent space within which you can navigate to identify the most important features for the classification.

In our example, we have intentionnaly biased the dataset by applying rotation of [0,90] degrees to cat images and rotation of [90,180] degrees to dog images.

Here is the added value of using the AAE model instead of a common classifier (Resnet34):
![latent spaces](https://github.com/LucaLaFisca/Human-Centered-xAI/blob/main/images/latent_spaces.png)
When navigating along the most discriminant dimension, we can clearly observe the effect of the rotation on the classification on the AAE latent space:
![rotation impact on AAE](https://github.com/LucaLaFisca/Human-Centered-xAI/blob/main/images/rotation_impact_xAAEnet.svg)
While, on the initial Resnet34 model, the effect of the rotation cannot be identified:
![rotation impact on Resnet34](https://github.com/LucaLaFisca/Human-Centered-xAI/blob/main/images/rotation_impact_Resnet34.svg)
