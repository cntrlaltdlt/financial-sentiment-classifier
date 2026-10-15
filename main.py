# main.py runs the program and ties all the components together

from financial_sentiment_classifier.dashboard import SentimentApp

if __name__ == "__main__":
    app = SentimentApp()
    app.mainloop()
