# Training Tutorial

## The MNIST Dataset
  
![Mnist Example 0](../img/mnist/img_11854_0.jpg)
![Mnist Example 1](../img/mnist/img_11253_1.jpg)
![Mnist Example 2](../img/mnist/img_10320_2.jpg)
![Mnist Example 3](../img/mnist/img_10324_3.jpg)
![Mnist Example 4](../img/mnist/img_40694_4.jpg)
![Mnist Example 5](../img/mnist/img_10596_5.jpg)
![Mnist Example 6](../img/mnist/img_19625_6.jpg)
![Mnist Example 7](../img/mnist/img_12452_7.jpg)
![Mnist Example 8](../img/mnist/img_10828_8.jpg)
![Mnist Example 9](../img/mnist/img_10239_9.jpg)



The MNIST dataset provides images of handwritten digits of 10 classes (0-9) and suits the task of simple image classification. 

The minimal MNIST arff file can be found in the `datasets/nominal` directory of the WekaDeeplearning4j package. This arff file lists all images in `datasets/nominal/mnist-minimal` and annotates their path with their class label.

**Important note:** The arff dataset contains two features, the first one being the `filename` and the second one being the `class`. Therefore it is necessary to define an `ImageDataSetIterator` in `Dl4jMlpFilter` or `Dl4jMlpClassifier` which uses these filenames in the directory given by the option `-imagesLocation`.

## GUI: LeNet MNIST Evaluation


The first step is to open the MNIST meta ARFF file in the Weka Explorer `Preprocess` tab via `Open File`. A randomly sampled MNIST dataset of 420 images is provided in the WekaDeeplearning4j package for testing purposes (`$WEKA_HOME/packages/wekaDeeplearning4j/datasets/nominal/mnist.meta.minimal.arff`). In the next step, the `Dl4jMlpClassifier` has to be selected as `Classifier` in the `Classify` tab. A click on the classifier will open the configuration window

![Classifier](../img/gui/mlp-classifier.png)

To correctly load the images it is further necessary to select the `Image-Instance-Iterator` as `instance iterator` and point it to the MNIST directory that contains the actual image files (`$WEKA_HOME/packages/wekaDeeplearning4j/datasets/nominal/mnist-minimal/`). 

![Image Instance Iterator](../img/gui/image-instance-iterator.png)

Select `LeNet` from the `zooModel` option as network architecture. 

![LeNet](../img/gui/layer-array.png)

A holdout evaluation strategy has to be selected in the `Test options` box via `Percentage split`, which can be set to 66% for a 2/3 - 1/3 split. The classifier training is now ready to be started with the `Start` button. The resulting classifier evaluation can be examined in the `Classifier output` box. Here, an evaluation summary is shown for the training and testing split. 

The above setup, trained for 50 epochs with a batch size of 256 produces a classification accuracy of 93.71% on the test data after training on the smaller sampled MNIST dataset and a score of 98.57% after training on the full MNIST dataset.

| MNIST   |  Train Size |  Test Size |  Train Accuracy |  Test Accuracy | Train Time      |
| -----   | ----------: | ---------: | --------------: | -------------: | --------------: |
| Sampled |         277 |        143 |          100.0% |         93.71% | 48.99s          |
| Full    |      60.000 |     10.000 |          98.76% |         98.57% | 406.30s         |

Table 1: Results for training time and classification accuracy after 50 epochs for both the sampled and the full MNIST training dataset using the LeNet architecture. Experiments were run on a NVIDIA TITAN X Pascal GPU.


## Commandline
The following run creates a Conv(3x3x8) > Pool(2x2,MAX) > Conv(3x3x8) > Pool(2x2,MAX) > Out architecture
```bash
$ java -Xmx5g weka.Run \
     .Dl4jMlpClassifier \
     -S 1 \
     -iterator ".ImageInstanceIterator -imagesLocation datasets/nominal/mnist-minimal -numChannels 1 -height 28 -width 28 -bs 16" \
     -normalization "Standardize training data" \
     -layer "weka.dl4j.layers.ConvolutionLayer -nFilters 8 -activation weka.dl4j.activations.ActivationReLU -columns 3 -rows 3 -paddingX 0 -paddingY 0 -strideX 1 -strideY 1" \
     -layer "weka.dl4j.layers.SubsamplingLayer -columns 2 -rows 2 -paddingX 0 -paddingY 0 -poolingType MAX -strideX 1 -strideY 1" \
     -layer "weka.dl4j.layers.ConvolutionLayer -nFilters 8 -activation weka.dl4j.activations.ActivationReLU -columns 3 -rows 3 -paddingX 0 -paddingY 0 -strideX 1 -strideY 1" \
     -layer "weka.dl4j.layers.SubsamplingLayer -columns 2 -rows 2 -paddingX 0 -paddingY 0 -poolingType MAX -strideX 1 -strideY 1" \
     -layer "weka.dl4j.layers.OutputLayer -activation weka.dl4j.activations.ActivationSoftmax -lossFn weka.dl4j.lossfunctions.LossMCXENT" \
     -config "weka.dl4j.NeuralNetConfiguration -updater weka.dl4j.updater.Adam" \
     -numEpochs 10 \
     -t datasets/nominal/mnist.meta.minimal.arff \
     -split-percentage 80
```

# Classify Your Own Dataset
  
![Plant Seedling 1](../img/plant-seedlings/0ac0f0a66.png)
![Plant Seedling 1](../img/plant-seedlings/3affdd752.png)

This tutorial will walk through the steps required to finetune a pretrained model on your custom dataset.

The dataset used in this tutorial is from the [Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification) competition on Kaggle. Visit the link to download the dataset.

### Creating the Meta ARFF File (`ImageDirectoryLoader`)
This dataset is uses a common 'folder organised' format - images are sorted into subfolders, with the class name being the subfolder name. This format is intuitive and easy to work with but cannot be loaded directly into WEKA without further processing.

WekaDeeplearning4j now comes with the `ImageDirectoryLoader`, a simple tool which creates an `.arff` file from a 'folder organised' dataset.

#### GUI Usage
The `ImageDirectoryLoader` can be invoked by selecting a **folder** instead of a **file** from the
`Open file...` menu.

Click `Open File...` and navigate to the `train/` folder in the Plant Seedlings dataset
you just downloaded.

![Image Directory](../img/gui/train-your-own-image-loader.png)

Click `Ok` and choose the `ImageDirectoryLoader` in the following popup.

![Image Directory](../img/gui/train-your-own-image-loader-2.png)

There are no settings to change so simply click `OK` to run - you should be taken back to the 
`Preprocess` panel with your instances now loaded.

![Image Directory](../img/gui/train-your-own-image-loader-complete.png)

#### Commandline Usage
The tool can also be run from the command line
```bash
java weka.Run .ImageDirectoryLoader -i <input dataset path> -name <output arff filename>
```
e.g.:
```bash
java weka.Run .ImageDirectoryLoader -i /path/to/plant-seedlings/data/train -name plant-seedlings-train.arff
```

The associated meta `.arff` file has been created inside the input directory specified. As we're simply checking accuracy within WEKA, we won't load in the `test/` data and submit it to Kaggle - that is outside the scope of this tutorial.

**Important note:** The newly created arff dataset contains two features, the first one being the `filename` and the second one being the `class`. Therefore it is necessary to define an `ImageInstanceIterator` (in `Dl4jMlpFilter` or `Dl4jMlpClassifier`) which uses these filenames in the directory given by the option `-imagesLocation`.


## Training - GUI

The first step is to open the plant seedlings instances as shown earlier. 
In the next step, the `Dl4jMlpClassifier` has to be selected as `Classifier` in the `Classify` tab. A click on the classifier will open the configuration window

![Classifier](../img/gui/mlp-classifier.png)

To correctly load the images it is further necessary to select the `Image-Instance-Iterator` as `instance iterator` 
and set the `images location` to the `train/` directory in the Plant Seedlings dataset folder.
We'll be using a pretrained model (which has a fixed input size) so the width, height, and number of channels don't need to be set. 

![Image Instance Iterator](../img/gui/train-your-own-iii.png)

For the sake of this tutorial, we'll use a pretrained Keras ResNet152V2 model. Select `KerasResNet` from the `zooModel` option.
The default variation is `RESNET50`, so we'll change that to `RESNET152V2`.

![Zoo Model Config](../img/gui/train-your-own-zooModel.png)

Note that by default, the layer specification is **not** loaded in the GUI for usability reasons;
loading the layers every time an option is changed can slow down the GUI significantly. If, however, you'd like
to view the layers of the zoo model you've selected, set the `Load layer specification in GUI` flag to true.

A holdout evaluation strategy has to be selected in the `Test options` box via `Percentage split`, 
which can be set to 66% for a 2/3 - 1/3 split. The classifier training is now ready to be started with the `Start` button. 
The resulting classifier evaluation can be examined in the `Classifier output` box. Here, an evaluation summary is shown for the training and testing split. 

The above setup, trained for 20 epochs with a batch size of 16 produces a classification accuracy of 94.51% on the
 test data.

```text

Correctly Classified Instances        1497               94.5076 %
Incorrectly Classified Instances        87                5.4924 %
Kappa statistic                          0.9392
Mean absolute error                      0.01  
Root mean squared error                  0.0894
Relative absolute error                  6.6502 %
Root relative squared error             32.5587 %
Total Number of Instances             1584     

=== Confusion Matrix ===

   a   b   c   d   e   f   g   h   i   j   k   l   <-- classified as
  44   0   0   0   6   0  38   0   0   0   0   0 |   a = Black-grass
   0 128   0   0   0   1   0   0   0   1   0   0 |   b = Charlock
   0   0  92   2   0   0   0   0   0   0   0   2 |   c = Cleavers
   0   1   0 202   0   0   1   0   0   0   0   0 |   d = Common Chickweed
   2   0   0   0  69   0   2   0   0   0   0   1 |   e = Common wheat
   1   0   1   0   1 154   0   0   0   0   0   1 |   f = Fat Hen
  11   0   0   0   0   0 206   0   1   0   0   0 |   g = Loose Silky-bent
   0   0   0   0   0   0   3  71   0   0   0   0 |   h = Maize
   1   0   0   0   0   0   0   0 171   0   0   0 |   i = Scentless Mayweed
   0   0   0   1   0   0   0   0   5  71   0   0 |   j = Shepherds Purse
   0   0   0   0   0   0   1   0   0   0 164   0 |   k = Small-flowered Cranesbill
   0   1   0   2   0   0   0   0   0   0   0 125 |   l = Sugar beet
```


## Training - Commandline
Ensure `weka.jar` is on the classpath. The following run finetunes a pretrained ResNet model for 20 epochs. This shows how to specify a non-default variation from the command line.
```bash
$ java weka.Run \
    .Dl4jMlpClassifier \
    -S 1 \
    -iterator ".ImageInstanceIterator -imagesLocation plant-seedlings/data/train -bs 16" \
    -normalization "Standardize training data" \
    -zooModel ".KerasResNet -variation RESNET152V2" \
    -config "weka.dl4j.NeuralNetConfiguration -updater \"weka.dl4j.updater.Adam -lr 0.1\"" \
    -numEpochs 20 \
    -t plant-seedlings/data/train/plant-seedlings-train.arff \
    -split-percentage 66
```
