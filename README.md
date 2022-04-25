# Project2ArtInt
Project 2 of Artificial Intelligence about movie recommender using Content-based filtering<br />
Please run the following commands to install the libraries that will be used in order to run the program<br />
pip install numpy<br />
pip install pandas<br />
pip install matplotlib<br />
pip install sklearn<br />
<br />
After installing these libraries, please run the run.bat file in the **command prompt**(cmd) in order to run the program. You can do so by simply running **run**
<br />
<br />
This project is intended to be used and tested in **Windows**, since it uses plots to show some graphics. However, if it wants to be tested in Linux, please follow the following:

1- pip3 install --upgrade pip --user <br />
2- Install the libraries<br />
pip install numpy<br />
pip install pandas<br />
pip install matplotlib<br />
pip install sklearn<br />
3- Comment the line 135 of main.py where it calls the plot. It would be the line containing "myMovieRecommender.plot(myMovieRecommender.moviesDF)"
4- run python3 main.py creditsDataset.csv moviesDataset.csv
