# super_advanced_AI

This is task 1. Aim of the task is to fit the mean value of the data set. 

## Installation and Development environment setup for Ubuntu 18.04 LTS 
Ensure you have python3 installed.
* Make sure your pip3 version (`pip3 --version`) is **19** or newer. The upgrade of the requirements will not work as expected with an older pip if the requirement version is specified using a git hash instead of a number.
* Install all required python-dev packages: `apt-get install python3.<x>-dev` with x either 5, 6 or 7 depending whether you use python 3.5, 3.6 or 3.7
* Clone this repo
* (Recommended) Install all dependencies to a python virtual environment as documented in the next session. 

#### Setting up virtual environment for Ubuntu 18.04 LTS:
You can follow theese steps to setup your virtual enviroments: 

* Install virtualenv: `pip3 install virtualenv`
* Create a directory where you want to store your environment variables for the project. E.g. `mkdir /home/wingtra/envs/sadab`
* Create your virtual environment: `virtualenv /home/wingtranaut/envs/sadab`
* Activate the virtual environment: `source /home/wingtranaut/envs/sadab/bin/activate`
* Install Cython: `pip3 install Cython`
* Now cd to sadab project directory and install dependencies by: `pip3 install -r requirements.txt`
* Use the SADAB analysis software as you like.
* To deactivate virtualenv: `deactivate`
* Activate your environment whenever using the SADAB analysis software, and deactivate when done. This is one of the robust ways to ensure your toolchain doesn't run into dependency conflicts with your other projects.

#### Running the Application:
Use the following command to start the application: `python3 main.py`
