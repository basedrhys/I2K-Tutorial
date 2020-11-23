# WEKA Setup / General Tips & Tricks

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

- `PATH`: The `bin/` folder of your Java installation should be appended to your `PATH` e.g., `C:\Program Files\Java\jdk1.8.0_271\bin`. If you can run the `java -version` command above then this is correctly set.
- `WEKA_HOME`: This variable should point to the location of your WEKA installation, e.g., `C:\Program Files\weka-3-8-4`.
- `CLASSPATH`: This variable is used by Java to locate classes to be preloaded. This can be manually specified for each command but is  abit arduous to do for every command, so it's recommended to set this permanently (at least for the duration of this tutorial). This variable should point to the location of `weka.jar` on your machine - typically inside the WEKA installation directory, e.g., `$env:WEKA_HOME\weka.jar`

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

The package can also be installed from the commandline 

#### CPU
For the package no further requisites are necessary.

#### GPU
The GPU additions needs the CUDA Toolkit 10.0, 10.1, or 10.2 backend with the appropriate cuDNN library to be installed on your system. Nvidia provides some good installation instructions for all platforms:

##### CUDA Toolkit
- [Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Mac OS X](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html)
- [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

##### CUDNN
- [Linux](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux)
- [Windows](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows)

## Add GPU Support

To add GPU support, [download](https://github.com/Waikato/wekaDeeplearning4j/releases/latest) and run the latest `install-cuda-libs.sh` for Linux/Macosx or `install-cuda-libs.ps1` for Windows. Make sure CUDA is installed on your system as explained [here](https://deeplearning.cms.waikato.ac.nz/install/#gpu).

The install script automatically downloads the libraries and copies them into your wekaDeeplearning4j package installation. If you want to download the library zip yourself, choose the appropriate combination of your platform and CUDA version from the [latest release](https://github.com/Waikato/wekaDeeplearning4j/releases/latest) and point the installation script to the file, e.g.:
```bash
./install-cuda-libs.sh ~/Downloads/wekaDeeplearning4j-cuda-10.2-1.6.0-linux-x86_64.zip
```
