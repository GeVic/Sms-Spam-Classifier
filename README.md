# Sms-Spam-Classifier
Over recent years, as the popularity of mobile phone devices has increased, SMS has grown into
a multi-million dollars industry. Which resulted in unsolicitated commercial advertisements
(spams) being sent to mobile devices. In parts of Asia, up to 30% of text messages were spam 
in 2012. 
In this project, a database from UCI and IIT Dehli is used, and after preprocessing and 
feature extraction, different machine learning algorithm are applied to the database.

```print("Work still in progress to introduce better result from current models")```

## Briefs of file
- ```spams.csv``` Dataset file in the preferred .csv format with IIT Delhi dataset being mergerd with.
- ```formatted_set.py``` To format the code into the required .csv and into particular columns.
- ```preprocessing.py``` Minmalistic code required for preprocessing, feature extraction which is to be performed on the dataset.
- ```README.md``` You are reading it now ;-)

## Database information
```As given on UCI website```

This corpus has been collected from free or free for research sources at the Internet: 

- A collection of 425 SMS spam messages was manually extracted from the Grumbletext Web site. This is a UK forum in which cell phone users make public claims about SMS spam messages, most of them without reporting the very spam message received. The identification of the text of spam messages in the claims is a very hard and time-consuming task, and it involved carefully scanning hundreds of web pages. The Grumbletext Web site is: [Web Link]. 
- A subset of 3,375 SMS randomly chosen ham messages of the NUS SMS Corpus (NSC), which is a dataset of about 10,000 legitimate messages collected for research at the Department of Computer Science at the National University of Singapore. The messages largely originate from Singaporeans and mostly from students attending the University. These messages were collected from volunteers who were made aware that their contributions were going to be made publicly available. The NUS SMS Corpus is avalaible at: [Web Link]. 
- A list of 450 SMS ham messages collected from Caroline Tag's PhD Thesis available at [Web Link]. 
- Finally, we have incorporated the SMS Spam Corpus v.0.1 Big. It has 1,002 SMS ham messages and 322 spam messages and it is public available at: [Web Link]. This corpus has been used in the following academic researches: 

Apart from the UCI corpus, IIT Delhi dataset also been mergerd with the same to increase the dataset and training. 

## Attribute information
The collection is just compose of one .csv file. It contains one message per line. Each line 
is composed by two columns: v1 contains the labels (ham or spam) and v2 contains the raw text.

## Citations
_If you find this dataset useful, you make a reference to our paper and the [web page](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/) in your papers, research, etc; 
Send us a message to talmeida ufscar.br or jmgomezh yahoo.es in case you make use of the corpus.
 
The SMS Spam Collection has been created by Tiago A. Almeida and José María Gómez Hidalgo._
