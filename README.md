# video_processing_playground
A test repository to play with image and video processing using python


I followed these articles to get set up:
https://realpython.com/traditional-face-detection-python/
https://realpython.com/face-detection-in-python-using-a-webcam/#pre-requisites

Two Methods to get things working:

Install Anaconda
https://docs.anaconda.com/anaconda/install/

Build and Enter conda environment:

I used python 3.7, even though it's old, because I was tired of encountering dependency issues in all the other methods I used to try to run open cv.
I ran into dependency issues, so tried python=3.12

$ conda create --name face-detection python=3.12
$ source activate face-detection

Then install dependencies:

conda install scikit-learn
conda install -c conda-forge scikit-image
conda install conda-forge::harfbuzz
conda install conda-forge::opencv

I had some issues with dependencies and this might solve them, but instead I found a version of open cv that didn't have issues putting this here as an alternative solution:
https://github.com/conda/conda-libmamba-solver/issues/348
conda config --set solver classic 

conda install scikit-learn && conda install -c conda-forge scikit-image && conda install conda-forge::harfbuzz && conda install -c menpo opencv3

OR

Use Pipenv to install dependencies:

pip3 install pipenv
pipenv install

Then change Vscode default interpreter to the newly created virtual env before running.