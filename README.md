# Identify music genre from an audio file

### Our purpose 💪

Music has always been both multi-faceted and deeply rooted in each cultural background. People all around the world have been making music long before the first orchestras ever appeared, and different ethnological and epochal influences contribute to produce completely unique and distinguishable musical qualities - classified as genres.

This classification, that has been evolving over time, is not always easy to achieve. Sometimes a genre can be hard to define, despite music of the same genre sharing similar properties such as tempo, beat, and rhythm. As such, devising an algorithm to classify music by its genre is a nigh impossible task. An efficient alternative to hard coding such program, is to use machine learning algorithms to accomplish this task. These Artificial Intelligence models are able to learn from data containing already classified music, and attempt to accurately determine the genre of further audio samples.

We used a Python script to create a data set of extracted musical features, and used that data to train three distinct machine learning algorithms R, for solving the classification task. We then analyzed and compared the accuracy of each of these three algorithms.

### Hot it works 🚀

1. Get a .wav file of a song of your choice.

2. Run the following command:
```
python3 Python_ExtractFeatures/inputSong.py <song.wav>
```
3. After running the previous command, a .csv file will be generated. After this, in the R_ClassificationTask/oneGenreGuess.R script change the path to the generated .csv as shown below:
```
# Import music values
musicValues <- read.csv("../Python_ExtractFeatures/song.csv", header=TRUE)
```
4. Run the R_ClassificationTask/oneGenreGuess.R script and view the results of the 3 models.

### Who we are 🧍

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/ruijramos">Rui Ramos</a></td>
            <td><a href="https://github.com/DuarteNRP">Duarte Pereira</a></td>
            <td><a href="https://github.com/RoninKingfisher">Isac Novo</a></td>
        </tr>
    </tbody>
</table>

### Used languages/frameworks 💻

<div>
  <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg" title="Python" alt="Python" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/r/r-original.svg" title="R" alt="R" width="40" height="40"/>&nbsp;
</div>

