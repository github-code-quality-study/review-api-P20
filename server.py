import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        self.valid_locations = set(review["Location"] for review in reviews)

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        query_params = parse_qs( environ.get("QUERY_STRING", "") )

        if environ["REQUEST_METHOD"] == "GET":
            filtered_reviews = reviews

            if "location" in query_params:
                location = query_params["location"][0]
                filtered_reviews = [ review for review in filtered_reviews if review["Location"] == location ]
            
            if "start_date" in query_params:
                start_date = datetime.strptime(query_params["start_date"][0], "%Y-%m-%d")
                filtered_reviews = [ review for review in filtered_reviews if datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") >= start_date ]
            
            if "end_date" in query_params:
                end_date = datetime.strptime( query_params["end_date"][0], "%Y-%m-%d" )
                filtered_reviews = [ review for review in filtered_reviews if datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") <= end_date ]
            
            for review in filtered_reviews:
                review["sentiment"] = self.analyze_sentiment(review.get("ReviewBody", ""))

            # Create the response body from the reviews and convert to a JSON byte string
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            content_length = int( environ.get("CONTENT_LENGTH", 0) )
            post_data_dict = parse_qs(environ["wsgi.input"].read(content_length).decode("utf-8"))

            location = post_data_dict.get("Location", [""])[0]
            review_body = post_data_dict.get("ReviewBody", [""])[0]

            if not review_body:
                start_response( "400 Bad Request", [("Content-Type", "application/json")] )
                return [b'{"error": "ReviewBody is missing."}']
            
            if not location:
                start_response( "400 Bad Request", [("Content-Type", "application/json")] )
                return [b'{"error": "Location is missing."}']
            
            if location not in self.valid_locations:
                start_response( "400 Bad Request", [("Content-Type", "application/json")] )
                response_body = json.dumps({"error": "invalid location"}, indent=2).encode("utf-8")
                return [response_body]
            
            review_id = str( uuid.uuid4() )
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            new_review = {
                "ReviewId": review_id,
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": timestamp
            }

            reviews.append(new_review)

            response_body = json.dumps(new_review, indent=2).encode("utf-8")

            start_response( "201 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ] )
            
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()