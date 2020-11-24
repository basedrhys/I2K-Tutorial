# Introduction & WEKA Setup

This is an introductory section which goes through the required steps to set up WEKA, **WekaDeeplearning4j**, and the GPU accelerated libraries (if your machine supports it).

At the end of this page are some short tips & tricks to make using WEKA even easier.

### Not Included in This Tutorial

**WekaDeeplearning4j** has a lot of features relating to deep-learning, but due to the limited scope of the tutorial, some of these were left out; this tutorial focuses solely on CNN-related techniques for image classification datasets. The following links may help you for other domains:

- RNN training: [WekaDeeplearning4j Docs](https://deeplearning.cms.waikato.ac.nz/examples/classifying-imdb)
- Image Segmentation: [ImageJ Docs](https://imagej.net/Trainable_Weka_Segmentation)

## Prerequisites
- Java 8 or above. To check, run `java -version` from the command line and verify you get an output something like the following (exact version numbers may differ):
    ```sh
    java version "1.8.0_271"
    Java(TM) SE Runtime Environment (build 1.8.0_271-b09)
    Java HotSpot(TM) 64-Bit Server VM (build 25.271-b09, mixed mode)
    ```
- Weka 3.8.4 or above ([download](https://sourceforge.net/projects/weka/files/latest/download))

## Environment Variables

The commandline examples in this tutorial assume a few environment variables are set. These can instead be manually specified for each command but this can be verbose so it's recommended to set them as permanent environment variables.

- `WEKA_HOME`: This variable should point to the location of your WEKA installation, e.g., `/home/rhys/weka-3.8.4`. It's used for some commands in this tutorial and by the `install-cuda-libs` script ([explained below](#wekadeeplearning4j-gpu-libraries)) for installing CUDA libraries for WEKA.
- `CLASSPATH`: This variable is used by Java to locate classes to be preloaded. This can be manually specified for each command but this becomes arduous to do every time, so it's recommended to set this permanently (at least for the duration of this tutorial). This variable should point to the location of `weka.jar` on your machine - typically inside the WEKA installation directory, e.g., `$WEKA_HOME\weka.jar`.

## Installing WekaDeeplearning4j

The final required step is to install the **WekaDeeplearning4j** package from the WEKA Package Manager.
- From the **Weka GUI Chooser**, open the **Package Manager** (`Tools` > `Package Manager`).
- Select `wekaDeeplearning4j` from the package list (near the bottom) and click `Install`. The package size is ~500mb so this may take a few moments.
- Restart WEKA to use the newly installed package.

The package can also be  simply via the commandline by downloading the most recent [package zip](https://github.com/Waikato/wekaDeeplearning4j/releases/latest):
```bash
$ java weka.core.WekaPackageManager -install-package <PACKAGE-ZIP>
```

You can check whether the installation was successful with
```bash
$ java weka.core.WekaPackageManager -list-packages installed
```
which results in
```
Installed	Repository	Loaded	Package
=========	==========	======	=======
1.7.0    	-----     	Yes	    <PACKAGE>: Weka wrappers for Deeplearning4j
```

## Add GPU Support

The GPU additions needs the CUDA Toolkit 10.0, 10.1, or 10.2 backend with the appropriate cuDNN library to be installed on your system. Nvidia provides some good installation instructions:

### CUDA Toolkit
- [Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

### CUDNN
- [Linux](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux)
- [Windows](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows)

### WekaDeeplearning4j GPU libraries

After setting up CUDA correctly on your machine, you'll need to download the WekaDeeplearning4j CUDA/CUDNN libraries. [Download](https://github.com/Waikato/wekaDeeplearning4j/releases/latest) and run the latest `install-cuda-libs.sh` for Linux or `install-cuda-libs.ps1` for Windows.

The install script automatically downloads the libraries and copies them into your wekaDeeplearning4j package installation. If you want to download the library zip yourself, choose the appropriate combination of your platform and CUDA version from the [latest release](https://github.com/Waikato/wekaDeeplearning4j/releases/latest) and point the installation script to the file, e.g.:
```bash
./install-cuda-libs.sh ~/Downloads/wekaDeeplearning4j-cuda-10.2-1.6.0-linux-x86_64.zip
```

## General Tips

### Running WEKA from the Command Line

A common workflow is to experiment with different models/hyperparameters in the **WEKA Explorer** on a small subset of the data,
then run the final configuration on a more powerful machine/server with the full dataset. Figuring out the correct command-line syntax can be difficult, especially for complex models, so WEKA has a `Copy configuration to clipboard` function.

1. Set up your configuration in the **WEKA Explorer** window, then right click and click `Copy configuration to clipboard`:
    
    ![Copy configuration to clipboard example](./images/1-copyConfigToClipboard.png)

2. Paste this into the command line (e.g., on your association's machine learning servers), specifying any other necessary flags run not included in the pasted configuration. For example training a `Dl4jMlpClassifier` can be done like:

    ```bash
    $ java weka.Run <Dl4jMlpClassifier configuration from clipboard> \
        -t <.arff file previously loaded into Weka> \
        -d <output model file path> \
        -split-percentage 80
    ```

