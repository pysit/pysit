# Installation

## Dependencies

PySIT has the following dependencies

    - Python 3.6 (or greater; So far we still need Python 2.7 to install the toolbox ––– PETSC)

    - NumPy 1.15 (or greater)

    - SciPy 1.1 (or greater)

    - matplotlib 2.2.3 (or greater)

    - PyAMG 4.0.0 (or greater)

    - ObsPy 1.1 (or greater)

    - PETSC4Py 3.6 (for MAC OS) and PETSC4Py 3.9 (for Linux)

External packages:

    - PETSC 3.6 (for MAC OS) and PETSC 3.9 (for Linux)

For optional parallel support, PySIT can depend on

    - MPI4Py 3.0.0


## Installing Python and PySIT Dependencies

On all platforms (Linux, Windows 7 or greater, and MAC OS X), we recommend a preassembled scientific python distribution, such as [Continuum IO’s Anaconda] or [Enthought’s Canopy]. These collections already include compatible (and in some cases accelerated) versions of most of PySIT’s dependencies. Download and follow the appropriate instructions for your operating system/distribution. In this instruction, we will show a step by step example that uses Miniconda3 to install PySIT.

### Step 1: Install Miniconda3

1. The easiest way to install Miniconda3 is to download it from the [download page of Miniconda]. Please select the Miniconda that works for your platform. There are two versions **Python 3.7** and **Python 2.7**. Please download the one with **Python 3.7** that corresponds to Miniconda3, while the one with **Python 2.7** corresponds to **Python 2.7**.

2. Download the proper installation file to your local directory, for example `~/Download` (In this example, we assume that the installation file of miniconda is located at `~/Download`. You can change it to any directory that you want). You will find a file named similar as `Miniconda3-latest-MacOSX-x86_64.sh` (for MAC OS users) or `Miniconda3-latest-Linux-x86_64.sh` (for Linux users) in your directory.

3. Open your terminal. Go to the directory `~/Download` by executing the  following command:

    ```sh
    $ cd ~/Download
    ```

    Then, install Miniconda3 by executing the following command for MAC OS users:

    ```sh
    $ source ./Miniconda3-latest-MacOSX-x86_64.sh
    ```

    and executing the following command for Linux useres:

    ```sh
    $ source ./Miniconda3-latest-Linux-x86_64.sh
    ```

    After the installation, you can check if Miniconda3 has been installed successfully by executing the command:

    ```sh
    $ which conda
    ```

    If it has been installed, then you will see the following output on the screen:

    ```
    $ /YOURHOMEDIRECTORY/miniconda3/bin/conda
    ```

### Step 2: Create a Python3 virtual environment by Miniconda3 and install PySIT

1. Create a Python3 virtual environment named with `myenv` with necessary packages by excuting the following command

    ```sh
    $ conda create -n myenv numpy=1.15.1 scipy=1.1.0 matplotlib=2.2.3 pyamg=4.0.0 
    ```

    This command will create a virtual Python environment with most of the dependencies.

2. Activate your environment by the following command:

    ```sh
    $ source activate myenv
    ```

    If the environment is created successfully, you would see the following sentence at the left bottom of your terminal:

    ```sh
    $(myenv)YourComputerName:
    ```

    where `YourComputerName` stands for the actual name of your computer.

3. Install ObsPy by pip. We assume that you already have **pip** installed. If not, please go to the [webpage of pip] to install **pip** first. With **pip** installed, you can install ObsPy by executing the following command:

    ```sh
    $ pip install obspy
    ```

4. Install necessary external softwares including [PETSC] and [MUMPS]. Assume that you clone or download the PySIT package at directory `/PATHTOPYSIT`. First go to the directory by executing the following command:

    ```sh
    $ cd /PATHTOPYSIT
    ```

    Then, execute the following commands to install PETSC and MUMPS. For linux users please use:

    ```sh
    $ source ./install_petsc4py_linux.sh
    ```

    For MAC OS users:

    ```sh
    $ source ./install_petsc4py_OSx.sh

    ```

5. Install the PySIT toolbox. First, make sure that you are in the PySIT directory by executing the following command:

    ```sh
    $ cd /PATHTOPYSIT
    ```

   Then, execute the following command:

   ```sh
   $ python setup.py install
   ```


6. Check if PySIT has been successfully installed. Please open a python by

   ```sh
   $ python
   ```

   Then try to import the PySIT toolbox by

   ```sh
   $ import pysit
   ```

   If there are no warnings or errors on the screen, then congratulations, you have successfully installed the PySIT toolbox. Please feel free to work with this powerful toolbox from the [examples].    





[Continuum IO’s Anaconda]: <https://www.anaconda.com/>
[Enthought’s Canopy]: <https://www.enthought.com/product/canopy/>
[download page of Miniconda]:<https://conda.io/miniconda.html>
[webpage of pip]:<https://pip.pypa.io/en/stable/installing/>
[PETSC]: <https://www.mcs.anl.gov/petsc/>
[MUMPS]: <http://mumps.enseeiht.fr/>
[examples]: <https://github.com/pysit/pysit/tree/master/examples>
