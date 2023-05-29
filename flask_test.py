from flask import Flask,jsonify
import joblib
from bs4 import BeautifulSoup
import pandas as pd
from urllib.request import urlopen, Request
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)


# @app.route("/")
# def index():
#     return render_template("index.html")


@app.route("/predict/<tick>", methods=["GET"])
def predict(tick):
    prediction = extractor(tick)
    pred = str(prediction[0])

    return jsonify({"data": pred})


def data_cleaning(text):
    text.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
    text["headline"] = text["headline"].str.lower()
    cleaned = " ".join(text["headline"])
    return cleaned


def sent_prediction(text):
    vader = SentimentIntensityAnalyzer()
    # Custom stock-related words
    new_words = {
        "revenue": 1.5,
        "earnings": 1.5,
        "profit": 1.5,
        "loss": -1.5,
        "acquisition": 1.5,
        "ipo": 1.5,
        "growth": 1.0,
        "expansion": 1.0,
        "layoffs": -1.5,
        "job cuts": -1.5,
        "partnership": 1.0,
        "product launch": 1.0,
        "regulation": -1.0,
        "lawsuit": -1.5,
        "settlement": 1.0,
        "scandal": -1.5,
        "investigation": -1.0,
        "competition": -1.0,
        "market share": 1.0,
        "guidance": 1.0,
        "forecast": 1.0,
        "quarterly": 1.0,
        "annual": 1.0,
        "earnings per share": 1.5,
        "revenue growth": 1.0,
        "cost cutting": 1.0,
        "debt": -1.0,
        "cash flow": 1.0,
        "valuation": 1.0,
        "stock split": 1.0,
        "share buyback": 1.0,
        "insider trading": -1.5,
        "stock options": 1.0,
        "volatility": -1.0,
        "index": 1.0,
        "trading": 1.0,
        "stock exchange": 1.0,
        "liquidity": 1.0,
        "blockchain": 1.0,
        "cryptocurrency": 1.0,
    }

    vader.lexicon.update(new_words)
    scores = vader.polarity_scores(text)
    scores1 = {
        "neg": [],
        "pos": [],
        "neu": [],
        "compound": [],
    }
    scores1["neg"].append(scores["neg"])
    scores1["neu"].append(scores["neu"])
    scores1["pos"].append(scores["pos"])
    scores1["compound"].append(scores["compound"])

    scores_df = pd.DataFrame(scores1)
    scores_df["Subjectivity"] = TextBlob(text).sentiment.subjectivity
    scores_df["Polarity"] = TextBlob(text).sentiment.polarity

    reg = joblib.load("model_jlib")
    predictions = reg.predict(scores_df)

    return predictions


def extractor(ticker):
    web_url = "https://finviz.com/quote.ashx?t="
    news_tables = {}
    tick = ticker

    url = web_url + tick
    req = Request(url=url, headers={"User-Agent": "Chrome"})
    response = urlopen(req)
    html = BeautifulSoup(response, "html.parser")
    # name = html.find(class_="text-blue-500").find("b").text
    news_table = html.find(id="news-table")
    news_tables[tick] = news_table
    news_list = []
    if(news_table):
        # Extracting first 25 news article
        count = 0
        for file_name, news_table in news_tables.items():
            for i in news_table.findAll("tr"):
                if count < 25:
                    count += 1
                    text = i.a.get_text()
                    date_scrape = i.td.text.split()
                    if len(date_scrape) == 1:
                        time = date_scrape[0]
                    else:
                        date = date_scrape[0]
                        time = date_scrape[1]
                    tick = file_name.split("_")[0]
                    news_list.append([tick, date, time, text])
                else:
                    break

        columns = ["ticker", "date", "time", "headline"]
        news_df = pd.DataFrame(news_list, columns=columns)

        # calling data cleaning

        clean_text = data_cleaning(news_df[["headline"]])

        return sent_prediction(clean_text)
    else:
        return 'N'


if __name__ == "__main__":
    app.run()
