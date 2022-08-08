from ariadne.server import Server
from ariadne.util import setup_logging
from ariadne.contrib.sklearn import SklearnMentionDetector
setup_logging()

server = Server()

# server.add_classifier("spacy_ner", SpacyNerClassifier("fr_ner4archives_default_vectors_lg"))
server.add_classifier("SklearnMentionDetector", SklearnMentionDetector())

app = server._app

# test

if __name__ == "__main__":
    server.start(debug=True, port=40022)
