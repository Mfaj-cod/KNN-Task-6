from modules.preprocess import preprocess_data
from modules.train_evaluate import train_evaluate

def main(data_path):
    x_train, x_test, y_train, y_test = preprocess_data(data_path=data_path)

    train_evaluate(x_train, x_test, y_train, y_test)


if __name__=="__main__":
    main('data/Iris.csv')