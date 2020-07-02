# Netflix-Recommendation-System

This repository contains the Python Code, using Pyspark dataframes, to build a Recommendation System. The underlying algorithm used is the Spark ALS(Alternating Least Squares) Method. 

DATA DESCRIPTION:
Data Source: 
The data that we have used to build the Recommendation Engine is from the famous Netflix Prize Open Competition, that the company held in 2006 to 2009 for the best algorithm to predict user ratings for films.
Link: https://www.kaggle.com/netflix-inc/netflix-prize-data

Data Size:  
5 GB (Four Customer Data Files and One Movie Data File)

Data Format and Explanation: 
Customer Data File Description:
Training Dataset consists of 4 text files in the following format:
MovieID:	These are the ID given to specific movies ranging from 1 to 17770 sequentially
CustomerID:	These are the ID given to specific customers ranging from 1 to 2649429, with gaps. There are 480189 users.	
Ratings:	These are on a five-star (integral) scale from 1 to 5	
TimeStamp:	These are Dates in the format YYYY-MM-DD.	Date Format

Movies File Description:
Movie information in "movie_titles.txt" is in the following format:</n>
MovieID:	These are the ID given to specific movies ranging from 1 to 17770 sequentially.
Movie_Title:	These are the Netflix movie title and may not correspond to titles used on other sites. Titles are in English.
YearOfRelease:	These can range from 1890 to 2005 and may correspond to the release of corresponding DVD, not necessarily its theatrical release.
