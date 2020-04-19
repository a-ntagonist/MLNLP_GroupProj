# Group Project for 2019 Fall Semester CS492: Machine Learning and Natural Language Processing

###Recreating [<Word embeddings quantify 100 years of gender and ethnic stereotypes>] (https://www.pnas.org/content/115/16/E3635) in Korean and Japanese

Group project repository for KAIST 2019 Fall Semester's Machine Learning and Natural Language Processing course. This code recreates the results in the PNAS paper <Word embeddings quantify 100 years of gender and ethnic stereotypes> with Korean and Japanese word embeddings.

* /code/: contains code for analysis.
  * regress.py: performs linear regression between real-life statistics and embedding bias, saves results to csv/txt files and draws plot
  * visu.py: uses PCA and t-SNE to plot key concepts and group vectors on 2-D space
  * gettop.py: gives top ten list of key concepts based on distances to group vectors and embedding bias
* /results/: contains resulting plots, csv tables and txt files
* /data/: excel files of statistics and lemmas

Word vector files should be saved in the /data/ directory to run the code.
