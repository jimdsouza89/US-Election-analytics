# US-Election-analytics
Identifying trends in the primaries of the US presidential elections, for Democrats and the GOP.

This code uses the Twitter Streaming API to collect tweets on the Democratic and Republican primaries
The tweets include that of parties, the candidates, and issues being discussed
The sentiment of the tweet is identified using a Random Forest classifier that was trained on a repository of 1million tweets (publicly available)
I also calculate the similarity of candidates, based on the public opinion of their stand on issues, their popularity, and their polling numbers
Finally the data is stored on a MySQL database, where it is later used by Tableau to create an interactive dashboard.
