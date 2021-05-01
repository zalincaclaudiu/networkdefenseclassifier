from flask import Flask
from flask import request
from DecisionTree import ClassificationTree
from DatasetUtils import getData


app = Flask(__name__)


@app.route('/checkRequest', methods=['POST'])
def index():
    content = request.json
    print(content["reqType"], content["url"], content["body"])
    return str(dtree.predictValue(content["reqType"], content["url"], content["body"]))



if __name__ == "__main__":
    xtrain, xtest, ytrain, ytest = getData()
    dtree = ClassificationTree()
    dtree.fit(xtrain, ytrain)
    ypred = dtree.predict(xtest)
    c = 0
    norm = 0
    anorm = 0
    for i in range(len(xtest)):
        if(int(ytest[i]) == int(ypred[i])):
            c += 1
            if(int(ypred[i])==0):
                norm += 1
            else:
                anorm += 1
    print('Accuracy: ' + str(c * 100 / len(xtest)))
    print('Norm: ' + str(norm))
    print('Anorm: ' + str(anorm))
    app.run(debug=True, port=4000)