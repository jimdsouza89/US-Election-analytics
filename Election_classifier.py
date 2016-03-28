'''
Created on 01-Sep-2015

@author: Jim D'Souza

Description : This code uses the Twitter Streaming API to collect tweets on the Democratic and Republican primaries
The tweets include that of parties, the candidates, and issues being discussed
The sentiment of the tweet is identified using a Random Forest classifier that was trained on a repository of 1million tweets (publicly available)
I also calculate the similarity of candidates, based on the public opinion of their stand on issues, their popularity, and their polling numbers
Finally the data is stored on a MySQL database, where it is later used by Tableau to create an interactive dashboard
'''

import traceback

import re
import nltk
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

#import sqlite3
import MySQLdb
import json
import csv

import httplib
import urllib3

from geopy.geocoders import Nominatim

import os
import sys
import random
import time
import datetime
#from openpyxl.xml.functions import tag

### Change file path according to the system being used ###
local_path = "D:\\US Election\\Model\\"
azure_path = "/home/brillio/twitterAnalysis/"
file_path = azure_path

tweet_file = file_path+"tweets.json"

dict_input = file_path+ "dict_output.csv"
#feature_file = file_path+ "feature_file.csv"
stopWordListFileName = file_path+ "Stopwords.csv"
punctuationListFileName = file_path+ "Punctuations.csv"
all_tags_file = file_path + "All_Tags.csv"
state_codes_file = file_path +"state_codes.csv"
decision_tree_file = file_path+ "decision_tree_file.joblib.pkl"
coordinatesFileName = file_path +"coordinates.csv"

tweets_data_path = file_path+ "twitter_data.csv"

### MySQL server credentials ###
table_name   = 'tweets_sentiments'
similarity_matrix = 'similarity_matrix'
db_localhost = "localhost"
db_user      = "twitter"
db_password  = "twitter"
db_name      = "sentiments"

### Pre-process the tweets
def processTweet(tweet):
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet

### Initialize stopWords ###
stopWords = []
featureList = []

### Check if the selected word is present in a tweet
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end
    
### Replaces multiple occurances of a word
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

### Create a stopWord list
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

### Create a feature vector, containing the counts of words present in the tweet, and removing stop words and punctuations
def getFeatureVector(tweet, stopWords):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#end

### Find n-grams in the text 
def find_ngrams(input_string,n):
    ngrams = nltk.ngrams(input_string,n)
    return ngrams

### Function to check if a word is present in the text
def findWholeWord(word, string, name=0):
    if name == 0 :
        if re.search(r"\b" + re.escape(word) + r"\b", string) :
            return True
        else :
            return False
        
    elif name == 1 :
        if re.search(word,string) :
            return True
        else :
            return False
    
### Creating the dataset that is to be used by the Random forest model later on
def create_model_dataset(word_file, ngrams, stopWords, Punctuations,model_tweet) :

    inpTweets = model_tweet
  
    feature_set = []
    rc = 0
    for row in inpTweets:
        feature_row = []
        feature_row.append(row[0])
        feature_row.append(row[1])
        feature_row.append(row[2])
        feature_row.append(row[3])
        if rc == 0 :
            feature_row.append("Stopwords")
            feature_row.append("Punctuations")
            feature_row.append("Negations")
            for val in ngrams :
                feature_row.append(val)
        elif rc <= 50000 :
            # Count stopwords
            stopwordsum = 0
            for val in stopWords:
                if findWholeWord(val,row[2]):
                    stopwordsum += 1    
            feature_row.append(stopwordsum)
            
            # Count punctuations
            punctuationsum = 0
            for val in Punctuations:
                if val in row[2]:
                    punctuationsum += 1
            feature_row.append(punctuationsum)
            
            # Count negations
            negationsum = 0
            for val in ["not","never","no"]:
                if findWholeWord(val,row[2]):
                    negationsum += 1
            feature_row.append(negationsum)
            
            # Count dictionary words
            for val in ngrams :
                if findWholeWord(val,row[2]):
                    feature_row.append(1)
                else :
                    feature_row.append(0)
        else :
            break
        feature_set.append(feature_row)  
        rc = rc + 1
    
    print "New file created  : ", datetime.datetime.now()
        
    return feature_set


### Initial check to see if the table exists in the MySql db
def checkTableExists(dbcon, tablename):
    dbcur = dbcon.cursor()
    dbcur.execute("""SHOW TABLES LIKE '"""+tablename+"""'""")
    result = dbcur.fetchone()
    if result:
        dbcur.close()
        return True
    else:
        dbcur.close()
        return False
    
### This function identifies candidates that are most similar to each other, based on
### the sentiments of the public towards topics being discussed by the candidates and
### their overall polling
def create_similarity_matrix():

    # Import coordinates, and create a dictionary containing each candidate as a key
    Candidate_Dict = {}
    with open(coordinatesFileName,'rb') as file:
        contents = csv.reader(file)
        for x in contents :
            Candidate_Dict[x[0]] = {}
            Candidate_Dict[x[0]]["LineX"] = x[1]
            Candidate_Dict[x[0]]["LineY"] = x[2]
            Candidate_Dict[x[0]]["CircleY"] = x[3]
            Candidate_Dict[x[0]]["Affiliate_Party"] = x[4]
            Candidate_Dict[x[0]]["Positive"] = 0
            Candidate_Dict[x[0]]["Negative"] = 0
            Candidate_Dict[x[0]]["Keywords"] = {}
    print "Coordinates imported"
    
    # Create a db connection - MySQLdb
    conn = MySQLdb.connect(host   = db_localhost, # your host, usually localhost
                user   = db_user, # your username
                passwd = db_password, # your password
                db     = db_name) # name of the data base
            
    c = conn.cursor()
    
    # The step creates a new table, which contains a matrix of candidates, and their similarity scores
    if checkTableExists(conn, similarity_matrix) :
        # Write to database - MySQLdb
        try :
            # Extract all the data from the tweet_sentiments table
            c.execute("""SELECT * FROM """+table_name)
            data = c.fetchall()
            print "Data imported" 
            
            # Process each row of the data, and add to the Candidate's record in the dictionary
            rowcount = 0
            for row in data :
                rowcount = rowcount+1
                print "Row : ", rowcount
                if row[6] != "" and row[6] in Candidate_Dict :
                    # Give the party name
                    Candidate_Dict[row[6]]["Affiliate_Party"] = row[5]
                
                    # Adding a count of Keywords
                    if row[9] not in Candidate_Dict[row[6]]["Keywords"] and row[9] != "None" :
                        Candidate_Dict[row[6]]["Keywords"][row[9]] = {}
                        Candidate_Dict[row[6]]["Keywords"][row[9]]["Positive"] = 0
                        Candidate_Dict[row[6]]["Keywords"][row[9]]["Negative"] = 0
                        if row[3] == "Positive" :
                            Candidate_Dict[row[6]]["Keywords"][row[9]]["Positive"] = 1
                        else :
                            Candidate_Dict[row[6]]["Keywords"][row[9]]["Negative"] = 1
                    elif row[9] in Candidate_Dict[row[6]]["Keywords"] and row[9] != "None" :
                        if row[3] == "Positive" :
                            Candidate_Dict[row[6]]["Keywords"][row[9]]["Positive"] += 1
                        else :
                            Candidate_Dict[row[6]]["Keywords"][row[9]]["Negative"] += 1
                    
                    # Adding a count of Positive and Negative tweets
                    if row[3] == "Positive" :
                        Candidate_Dict[row[6]]["Positive"] += 1
                    else : 
                        Candidate_Dict[row[6]]["Negative"] += 1
            
            # Calculating the similarity score for each candidate by iterating through the keys of the dictionary
            for candidate in Candidate_Dict :
                Candidate_Dict[candidate]["Similarity_Score"] = 0
                
                # Get the top 3 keywords for the candidate
                max_1 = 0
                max_2 = 0
                max_3 = 0
                max_1_keyword = ""
                max_2_keyword = ""
                max_3_keyword = ""
                for keyword in Candidate_Dict[candidate]["Keywords"] :
                    if Candidate_Dict[candidate]["Keywords"][keyword] > max_1 :
                        max_3 = max_2
                        max_3_keyword = max_2_keyword
                        max_2 = max_1
                        max_2_keyword = max_1_keyword
                        max_1 = Candidate_Dict[candidate]["Keywords"][keyword]
                        max_1_keyword = keyword
                    elif Candidate_Dict[candidate]["Keywords"][keyword] > max_2 :
                        max_3 = max_2
                        max_3_keyword = max_2_keyword
                        max_2 = Candidate_Dict[candidate]["Keywords"][keyword]
                        max_2_keyword = keyword
                    elif Candidate_Dict[candidate]["Keywords"][keyword] > max_3 :
                        max_3 = Candidate_Dict[candidate]["Keywords"][keyword]
                        max_3_keyword = keyword

                print max_1_keyword, max_1
                print max_2_keyword, max_2
                print max_3_keyword, max_3
                
                # Iterate through all the remaining candidates in the Candidate Dictionary, and calculate the similarity score between Candidate 1 and 2
                for candidate_comp in Candidate_Dict :
                    Candidate_Dict[candidate]["Similarity_Score"] = 0.0
                    if candidate == candidate_comp :
                        Candidate_Dict[candidate]["Similarity_Score"] = 6.0
                    else :                                                
                        if Candidate_Dict[candidate]["Affiliate_Party"] == Candidate_Dict[candidate_comp]["Affiliate_Party"] :
                            Candidate_Dict[candidate]["Similarity_Score"] += 1.0
                        # Give each pair a positive sentiment score of (1- difference in positive tweet percentage)
                        if (Candidate_Dict[candidate]["Positive"]+Candidate_Dict[candidate]["Negative"]) == 0 :
                            candidate_pos = 0.0
                        else :
                            candidate_pos = (Candidate_Dict[candidate]["Positive"])/(Candidate_Dict[candidate]["Positive"]+Candidate_Dict[candidate]["Negative"])
                    
                        if (Candidate_Dict[candidate_comp]["Positive"]+Candidate_Dict[candidate_comp]["Negative"]) == 0 :
                            candidate_comp_pos = 0.0
                        else :
                            candidate_comp_pos = (Candidate_Dict[candidate_comp]["Positive"])/(Candidate_Dict[candidate_comp]["Positive"]+Candidate_Dict[candidate_comp]["Negative"])
                        Candidate_Dict[candidate]["Similarity_Score"] += (1.0 - abs(candidate_pos - candidate_comp_pos))
                        print "Positive tweet score : ", Candidate_Dict[candidate]["Similarity_Score"]
                        
                        # Give each pair a popularity score of ( 1 - Candidate1Tweets/Candidate2Tweets) [The ratio should be <1, else invert] 
                        if (Candidate_Dict[candidate]["Positive"]+Candidate_Dict[candidate]["Negative"]) == 0 or (Candidate_Dict[candidate_comp]["Positive"]+Candidate_Dict[candidate_comp]["Negative"]) == 0 :
                            tweet_score = 0.0
                        else :
                            tweet_score = (Candidate_Dict[candidate]["Positive"]+Candidate_Dict[candidate]["Negative"])/(Candidate_Dict[candidate_comp]["Positive"]+Candidate_Dict[candidate_comp]["Negative"])
                        if tweet_score >= 1.0 :
                            Candidate_Dict[candidate]["Similarity_Score"] += (1.0/tweet_score)
                        else :
                            Candidate_Dict[candidate]["Similarity_Score"] += (tweet_score)
                        print "Tweet count score : ", Candidate_Dict[candidate]["Similarity_Score"]
                        
                        # Get the top 3 keywords for the comparison candidate
                        max_1_comp = 0
                        max_2_comp = 0
                        max_3_comp = 0
                        max_1_comp_keyword = ""
                        max_2_comp_keyword = ""
                        max_3_comp_keyword = ""
                        for keyword in Candidate_Dict[candidate]["Keywords"] :
                            if Candidate_Dict[candidate]["Keywords"][keyword]["Positive"] == 0 or (Candidate_Dict[candidate]["Keywords"][keyword]["Positive"]+Candidate_Dict[candidate]["Keywords"][keyword]["Negative"]==0) :
                                pos_perc = 0
                            else :
                                if (Candidate_Dict[candidate]["Positive"]+Candidate_Dict[candidate]["Negative"]) > 0 :
                                    if (Candidate_Dict[candidate]["Keywords"][keyword]["Positive"]+Candidate_Dict[candidate]["Keywords"][keyword]["Negative"])/(Candidate_Dict[candidate]["Positive"]+Candidate_Dict[candidate]["Negative"]) > 0.05 :
                                        pos_perc = Candidate_Dict[candidate]["Keywords"][keyword]["Positive"]/(Candidate_Dict[candidate]["Keywords"][keyword]["Positive"]+Candidate_Dict[candidate]["Keywords"][keyword]["Negative"])
                                    else:
                                        pos_perc = 0
                            if pos_perc > max_1 :
                                max_3_comp = max_2_comp
                                max_3_comp_keyword = max_2_comp_keyword
                                max_2_comp = max_1_comp
                                max_2_comp_keyword = max_1_comp_keyword
                                max_1_comp = pos_perc
                                max_1_comp_keyword = keyword
                            elif Candidate_Dict[candidate]["Keywords"][keyword] > max_2 :
                                max_3_comp = max_2_comp
                                max_3_comp_keyword = max_2_comp_keyword
                                max_2_comp = pos_perc
                                max_2_comp_keyword = keyword
                            elif Candidate_Dict[candidate]["Keywords"][keyword] > max_3 :
                                max_3_comp = pos_perc
                                max_3_comp_keyword = keyword
                    
                        if max_1_keyword != "" and max_1_keyword == max_1_comp_keyword or max_1_keyword == max_2_comp_keyword or max_1_keyword == max_3_comp_keyword :
                            Candidate_Dict[candidate]["Similarity_Score"] += 1
                        if max_2_keyword != "" and max_2_keyword == max_1_comp_keyword or max_2_keyword == max_2_comp_keyword or max_2_keyword == max_3_comp_keyword :
                            Candidate_Dict[candidate]["Similarity_Score"] += 1
                        if max_3_keyword != "" and max_3_keyword == max_1_comp_keyword or max_3_keyword == max_2_comp_keyword or max_3_keyword == max_3_comp_keyword :
                            Candidate_Dict[candidate]["Similarity_Score"] += 1
                    
                    print "Score has been calculated for ", candidate," with ", candidate_comp," : ", Candidate_Dict[candidate]["Similarity_Score"]
                    
                    c.execute("""UPDATE """+similarity_matrix+""" SET Similarity_Score = %s WHERE Candidate1 LIKE %s AND Candidate2 LIKE %s;""", (str(Candidate_Dict[candidate]["Similarity_Score"]),candidate,candidate_comp))

            print "All records updated"  
            conn.commit()
            print "Commit"
        except :
            print "Not committed"
            conn.rollback()
    else :
        c.execute('''CREATE TABLE '''+similarity_matrix+''' 
            (NodeName text, Affiliate_Party text, Candidate1 text, Candidate2 text, Relationship text, 
            LineX text, LineY text, CircleY text, Similarity_Score text)''')
        
        for candidate in Candidate_Dict :
            for candidate_comp in Candidate_Dict :
                relationship = candidate+" --> "+candidate_comp
                c.execute('''INSERT INTO '''+similarity_matrix+''' VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);''', (candidate,Candidate_Dict[candidate]["Affiliate_Party"],candidate,candidate_comp,relationship,str(Candidate_Dict[candidate]["LineX"]),str(Candidate_Dict[candidate]["LineY"]),str(Candidate_Dict[candidate]["CircleY"]),str(1.0),))        
                conn.commit()
                c.execute('''INSERT INTO '''+similarity_matrix+''' VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);''', (candidate_comp,Candidate_Dict[candidate_comp]["Affiliate_Party"],candidate,candidate_comp,relationship,str(Candidate_Dict[candidate_comp]["LineX"]),str(Candidate_Dict[candidate_comp]["LineY"]),str(Candidate_Dict[candidate_comp]["CircleY"]),str(1.0),))        
                conn.commit()
                
    c.close()

### Checking if the tweet data is correctly classified
def verify_tweet(tweet_content,user,all_tags):
    
    Party = ""
    HD = "" 
    HT = ""
    KW = ""
    Candidate = ""
    
    for tag in all_tags :
        if findWholeWord(tag[0].lower(), tweet_content.lower(), name=1):
            if tag[1] == "Hashtag" :
                if HT == "":
                    HT = tag[0]
                    if Party == "" :
                        if tag[2] == "Democrat":
                            Party = "Democrat"
                        elif tag[2] == "Republican" :
                            Party = "Republican"
            elif tag[1] == "Handle" :
                if HD == "" :
                    HD = tag[0]
                    if Party == "" :
                        if tag[2] == "Democrat":
                            Party = "Democrat"
                        elif tag[2] == "Republican" :
                            Party = "Republican"
            elif tag[1] == "Keyword" :
                if KW == "":
                    KW = tag[0]
                    if Party == "" :
                        if tag[2] == "Democrat":
                            Party = "Democrat"
                        elif tag[2] == "Republican" :
                            Party = "Republican"
            elif tag[1] == "Candidate" :
                Candidate = tag[0]
                if tag[2] == "Democrat":
                    Party = "Democrat"
                elif tag[2] == "Republican" :
                    Party = "Republican"

            
            if tag[3] != "" :
                Candidate = tag[3]
                if Candidate != "":
                    if tag[2] == "Democrat":
                        Party = "Democrat"
                    elif tag[2] == "Republican" :
                        Party = "Republican"
            
    return Party, HD, HT, KW, Candidate

### This function takes in the model data set and classifies the tweets as positive or negative
def classify_tweet(data):
        
    try :
        
        decoded = json.loads(data)
        tweeter_location = ""
        
        if decoded["created_at"] != None :
            datetime = decoded["created_at"]
        else :
            datetime = datetime.datetime.now()
                        
        # Extract tweet content and user name
        tweet_content = decoded['text'].encode('ascii', 'ignore')
        user = decoded['user']['screen_name']
        
        # Import All Tags 
        all_tags = []
        rc = 0
        with open(all_tags_file,'rb') as file:
            contents = csv.reader(file)
            for x in contents :
                if rc != 0 :
                    all_tags.append(x)
                rc = rc + 1
        print "All Tags imported"
            
        Party, HD, HT, KW, Candidate = verify_tweet(tweet_content,user,all_tags)
            
        #if Party != "" or HD != "" or HT != "" or KW != "" or Candidate != "" :
        if Party != "" or HD != "" or HT != "" or Candidate != "" :

            # Import the RF model
            print ("Importing model file...")
            clf = joblib.load(decision_tree_file)
    
            # Import dictionary to add as columns  
            ngrams = []
            with open(dict_input,'rb') as file:
                contents = csv.reader(file)
                for x in contents :
                    ngrams.append(x[0])
            print "Dictionary imported"
    
            # Import stop words  
            stopWords = []
            with open(stopWordListFileName,'rb') as file:
                contents = csv.reader(file)
                for x in contents :
                    stopWords.append(x[0])
            print "Stop Words imported"
    
            # Import punctuations 
            Punctuations = []
            with open(punctuationListFileName,'rb') as file:
                contents = csv.reader(file)
                for x in contents :
                    Punctuations.append(x[0])
            print "Punctuations imported"
  
    
            # Import State codes
            State_codes = []
            with open(state_codes_file,'rb') as file:
                contents = csv.reader(file)
                for x in contents :
                    State_codes.append(x)
            print "State Codes imported"
            
            if decoded['place'] == None :
                tweeter_location = "Unknown"
            elif str(decoded['place']['country_code']) != "US" :
                tweeter_location = "Outside US"
            else :
                for state in State_codes :
                    tweeter_location = str(decoded['place']['full_name'])
                    if findWholeWord(state[0], tweeter_location) or findWholeWord(state[1], tweeter_location) or findWholeWord(state[2], tweeter_location) or findWholeWord(state[3], tweeter_location) or findWholeWord(state[4], tweeter_location):
                        tweeter_location = state[0]

            model_tweet = [['User','Datetime','Tweet','Sentiment']]
            model_tweet.append([user,datetime,tweet_content,''])
            
            # Prepare the tweet according to the data set required to classify the tweet, then store in a numpy array
            model_tweet = create_model_dataset(dict_input,ngrams,stopWords,Punctuations,model_tweet)
            model_tweet = np.asarray(model_tweet)
                
            tweet_feature = model_tweet[1,4:]
            
            # Use the model to generate a prediction for the tweet
            preds = clf.predict(tweet_feature)
            if preds[0] == 0 :
                pred = "Negative"
            else :
                pred = "Positive"
            print "Prediction : ", pred
            
            outputrow = []
            outputrow.append(str(user))                                                # User name
            outputrow.append(str(datetime))                                            # Date time
            outputrow.append(str(tweet_content))                                       # Tweet text
            outputrow.append(pred)                                                 # Sentiment prediction
            
            outputrow.append(tweeter_location)                                         # Location
            outputrow.append(Party)                                                    # Party
            outputrow.append(Candidate)                                                # Candidate
            outputrow.append(HT)                                                       # Hashtags
            outputrow.append(HD)                                                       # Handle
            outputrow.append(KW)                                                       # Sentiment prediction
                    
            
            # Write to csv file
            #print "Writing to csv"
            #with open(tweets_data_path, 'ab') as op:
            #    writer = csv.writer(op, delimiter=',')
            #    writer.writerow(outputrow)
            
            
            # Create a db connection - MySQLdb
            conn = MySQLdb.connect(host   = db_localhost, # your host, usually localhost
                        user   = db_user, # your username
                        passwd = db_password, # your password
                        db     = db_name) # name of the data base
        
            c = conn.cursor()

            # If table does not exist, create a new table. Otherwise, write to DB
            if checkTableExists(conn, table_name) :
                # Write to database - MySQLdb
                try :
                    c.execute('''INSERT INTO '''+table_name+''' VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);''', (outputrow[0],str(outputrow[1]),outputrow[2],outputrow[3],str(outputrow[4]),str(outputrow[5]),str(outputrow[6]),str(outputrow[7]),str(outputrow[8]),str(outputrow[9]),))        
                    conn.commit()
                    print "Commit"
                except :
                    print "Not committed"
                    conn.rollback()
            else :
                c.execute('''CREATE TABLE '''+table_name+''' 
                    (User text, Timestamp text, Tweet text, Sentiment text, Location text, Affiliate_Party text, Candidate text, Hashtags text, Handles text,
                    Keywords text)''')

            c.close()
            
    except Exception, err:
        print(traceback.format_exc())
 
    #except BaseException, e:
    #    #print 'failed on_data,', str(e)
    #    print "Base Exception"
    #    time.sleep(1)
    #    pass
    #except:
    #    # Oh well, reconnect and keep trucking
    #    print "Unable to access files"
    #    pass
        

### Main function
def main():

    start_time = time.time() #grabs the system time
    start = datetime.datetime.now()
    print "Starting time : ", datetime.datetime.now()        
    
    # Read in csv file containing temp tweets
    inpTweets = []
    fp = open(tweet_file, 'r')
    line = fp.readline()
    while line:
        txt = line.strip()
        inpTweets.append(txt)
        line = fp.readline()
    fp.close()
    #create_similarity_matrix()   
    
    for data in inpTweets :
        classify_tweet(data)
        
    
if __name__ == '__main__':
        
    main()
