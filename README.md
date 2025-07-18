# Analyzing the Dynamics of User Interactions with AI-Virtual vs. Human Influencers

## Overview
Artificial Intelligence (AI) is gaining popularity in social media, notably through the rise of AI Virtual Influencers (AIVIs) that act like and resemble humans.
While some evidence suggests AIVIs generate more engagement than human influencers (HIs), others report the opposite, revealing contradictory effects on consumer behavior.
Analysis of 156,918 Instagram posts and corresponding comments, encompassing 12,460,042 likes, indicating user-influencer interactions, was conducted to learn and compare engagement patterns with AIVI and HI.
Results show that AIVIs receive significantly less attention compared with HI, expressed by fewer likes, shorter comments, and more neutral sentiment in comments.
Utilizing user interaction patterns with influencers, we developed a machine learning model to classify influencers as either AIVIs or HIs, achieving an F1-score of 0.89.
This study advances the understanding of AIVIs in social media and provides a foundation for optimizing their use in digital engagement strategies.
The findings show that, despite their novelty, AIVIs are less effective at engaging audiences than HIs.

## Data Availability

Due to Instagram’s content sharing policies, we cannot share the raw dataset. 
However, a list of publicly available Instagram post URLs used in this study is provided to enable data reconstruction for reproducibility purposes.

## Running the code
`data_analysis.R` - 
This R script performs sentiment and popularity analysis on Instagram comment data, comparing interactions between users and two types of influencers: human (HI) and AI-based (AIVI).
Cleans comment text by removing usernames,
Applies sentiment analysis,
Computes comment length,
Performs statistical comparisons across groups (HI vs. AIVI) for:
Popularity,
Likes per post,
Likes per comment,
Sentiment scores,
Comment length, and
Visualizes results

Output: A CSV file with sentiment results and several plots highlighting group differences.


`EP_analysis.py` -
Analysis of Estimated Earnings per Post

`Neural_Net.py` -
Loads a JSON file containing Instagram comment features and BERT-style sentence embeddings.
Cleans and prepares the dataset:
Unpacks embedded vectors,
One-hot encodes sentiment,
Label-encodes used type (AIVI vs. HUMAN),
Trains a feed-forward neural network (MLPClassifier) to predict whether a comment was made in response to an AI or human influencer,
Evaluates the model using accuracy, classification report, confusion matrix, and ROC AUC.

`Text_to_Vector.py` -
Loads a JSON file (output_Final.json) containing user comments (likely scraped from Instagram).
Starts processing at a given index (start_index = 150000) in case of a crash or to resume processing.
Uses OpenAI’s `text-embedding-3-large` model to generate embeddings from user comment text (posts.comments.text).
Stores the result in a new column: embedded.posts.comments.text.
Implements retrying failed embedding requests with backoff.
Periodically saves output to JSON files named by index (OutPutUntilXXXX.json), plus a final export.

`graph_analysis.R` - 
Reads subgraph edge lists and builds directed igraph graphs.
Computes multiple centrality metrics (Indegree, Outdegree, Closeness, etc.).
Applies log transformation to normalize skewed centrality distributions.
Performs Wilcoxon tests to compare groups (AIVI vs. HI).
Visualizes the results using ggplot2 boxplots with overlayed mean (blue dots) and median (red triangles).
Annotates significance (p-values) on each facet.

`model.R` - 
Train classification models (Logistic Regression and Neural Networks) to differentiate between user interactions with AI-based Virtual Influencers (AIVIs) and Human Influencers (HIs) based on comment data, metadata, sentiment, and vector embeddings.
Data Preparation & Sentiment Analysis:
Loads Excel/CSV comment data.
Cleans usernames and calculates sentiment using sentimentr.
Computes derived features like comment length and popularity.
Saves enriched data.
Statistical Testing (t-tests and Wilcoxon):
Tests for group differences in popularity, likes per post/comment, text length, and sentiment.
Visualizes these differences with ggplot2 boxplots and overlays mean/error bars.

Logistic Regression with caret:
Trains a logistic regression model on selected features (likes_count, popularity, sentiment).
Evaluates performance via confusion matrix, F1 score, and ROC/AUC.
Exploratory Model with BERT and TF-IDF:
Includes textEmbed for BERT embeddings and tm for TF-IDF vectorization.
Trains logistic regression using comment text.
Makes predictions and evaluates performance.

`df2Neo4j.ipynb` -
Create a Neo4j graph object from comments using LangChain and LlaMA.
Then, create the two subgraphs of AIVI and HI.

## Miscellaneous
Please send any questions you might have about the code and/or the algorithm to alon.bartal@biu.ac.il.


## Citing
If you find this paper useful for your research, please consider citing us:
```
@article{j2025AIVI,
  title={User Engagement with Human vs. AI Influencers: A Social Media Behavior Analysis},
  author={},
  journal={Computers in Human Behavior},
  volume={},
  number={},
  pages={},
  year={2025},
  publisher={}
}
```


