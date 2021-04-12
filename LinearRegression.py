import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data1 = pd.read_csv("kc_house_data.csv")


def preprocess (data):
    data = data.drop(["long","lat"],axis = 1)
    data = data.dropna(axis = 0)
    zipcode = data["zipcode"]
    data = data.drop(["id","zipcode","date","sqft_living"],axis =1)
    set_zipcode = set(zipcode)
    processed = data.copy()
    for zip in set_zipcode:
        processed[zip] = [1 if rec_zip == zip else 0 for rec_zip in zipcode]
    processed = processed.drop(processed[processed.bedrooms == 0].index)
    processed = processed.drop(processed[processed.bathrooms == 0].index)
    return  processed

def main(data):
    x = range(1,100)
    processed_data = preprocess(data)
    train_losses= []
    test_losses = []
    for i in range (1,100):
        train_loss, test_loss = lin_reg_precentage(i,processed_data)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    plt.scatter(x,np.log(train_losses),label= "Train Loss")
    plt.scatter(x,np.log(test_losses),label = "Test Loss")
    plt.title("Train and Test loss per data percentage taken"
              " to be train")
    plt.xlabel("Percentage of data taken to be train")
    plt.ylabel("Log loss")
    plt.legend()
    plt.show()


def lin_reg_precentage (i,processed_data):
    precentage = i / 100
    len_of_train = int(precentage * len(processed_data))
    len_of_test = len(processed_data) - len_of_train
    train = processed_data.sample(len_of_train)
    test = processed_data.drop(train.index)
    y_train = train["price"]
    train = train.drop("price", axis=1)
    y_test = test["price"]
    test = test.drop("price", axis=1)
    w = linear_reg(np.transpose(train), y_train)
    train_loss = ((np.linalg.norm((np.linalg.multi_dot([w,np.transpose(train)])-y_train)))**2)/len_of_train
    test_loss = ((np.linalg.norm((np.linalg.multi_dot([w,np.transpose(test)])-y_test)))**2)/len_of_test
    return train_loss,test_loss

def linear_reg(traintr,y):
    dagger_transpose = np.transpose(np.linalg.pinv(traintr))
    return (np.matmul(dagger_transpose,y))

main(data1)