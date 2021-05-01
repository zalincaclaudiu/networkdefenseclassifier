from urllib.parse import unquote
import re
from sklearn.model_selection import train_test_split
import numpy as np
normal_path = 'dataset/normalTrafficTraining.txt'
anom_path = 'dataset/anomalousTrafficTest.txt'

def getFeatures(reqType, request):
    parts = request.split('\n')
    url1  = unquote(unquote(parts[0].split(" ")[0]))
    # print(parts)
    # print("/////////////////////////////")
    maxByte = max([ord(c) for c in request])
    urlParts = url1.split("/", 3)
    if(len(urlParts) == 4):
        url = urlParts[-1]
    else:
        url = ''
    body = parts[-3]
    if (len(parts[-4])==0 and len(parts[-2])==0):
        return [len(reqType)] + analyzeUrl(url) + analyzeBody(parts[-3])
    else:
        return [len(reqType)] + analyzeUrl(url) + analyzeBody('')

def analyzeUrl(url):
    symbols = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
    anomSymbols = "\'*\"<>/:?."
    numbers = 0
    letters = 0
    spaces = 0
    nrOfSpecialChars = 0
    nrAnomChars = 0
    for c in url:
        if(c.isdigit()):
            numbers += 1
        elif(c.isalpha()):
            letters += 1
        elif(c.isspace()):
            spaces += 1
        elif(c in symbols):
            nrOfSpecialChars += 1
        if(c in anomSymbols):
            nrAnomChars += 1

    urlLength = len(url)

    return [numbers, letters, spaces, nrOfSpecialChars, nrAnomChars, urlLength]

def analyzeBody(body):
    if(len(body)==0):
        return [0,0,0,0,0]
    symbols = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
    anomSymbols = "\'*\"<>/:?."
    numbers = 0
    letters = 0
    spaces = 0
    nrOfSpecialChars = 0
    nrAnomChars = 0
    for c in body:
        if (c.isdigit()):
            numbers += 1
        elif (c.isalpha()):
            letters += 1
        elif (c.isspace()):
            spaces += 1
        elif (c in symbols):
            nrOfSpecialChars += 1
        elif (c in anomSymbols):
            nrAnomChars += 1

    urlLength = len(body)
    return [nrOfSpecialChars, letters, numbers, nrAnomChars, urlLength]

def getData():
    fnorm = open(normal_path, "r")
    normText = fnorm.read()
    normRequests = re.split(r'(GET|POST|DELETE|PATCH)\s', normText)

    fanom = open(anom_path, "r")
    anomText = fanom.read()
    anomRequests = re.split(r'(GET|POST|DELETE|PATCH)\s', anomText)

    X = []
    Y = []

    for i in range(1, len(anomRequests), 2):
        features = getFeatures(anomRequests[i], anomRequests[i + 1])
        if (len(features) > 0):
            Y.append(1)
            # print(getFeatures(normRequests[i],normRequests[i+1]))
            X.append(features)

    for i in range(1, len(normRequests), 2):
        features = getFeatures(normRequests[i], normRequests[i + 1])
        if (len(features) > 0):
            Y.append(0)
            # print(getFeatures(normRequests[i],normRequests[i+1]))
            X.append(features)

    X = np.array(X)
    Y = np.array(Y)

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.22, shuffle=True)

    return xtrain, xtest, ytrain, ytest

def extractFeatures(reqType, url, body):
    return [len(reqType)] + analyzeUrl(url) + analyzeBody(body)

def shuffleData(X, y, seed):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffleData(X, y, seed)
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test