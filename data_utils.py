'''
Helper function to aid the understanding the titanic data set
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam

def getLettersAndNumbers(cabin):
    '''
    Function which is able to extract the letters and numbers from a given
    series of cabin names

    outputs 2 matrices of letters and numbers
    '''
    def no_num(x):
        num = ['0','1','2','3','4','5','6','7','8','9']
        if len(x) == 1 and x not in num:
            # print('no number')
            return True
    def is_num(x):
        num = ['0','1','2','3','4','5','6','7','8','9']
        if x in num:
            return True
        return False

    letters_matrix = []
    numbers_matrix = []
    for test in cabin:
        temp = test.split(' ')
        letters = []
        numbers = []
        for j in range(len(temp)):
            if temp[j] == 'Unknown':
                letter = 'U'
                number = '0'
            for i in range(len(temp[j])):
                if no_num(temp[j][i]):
                    letter = temp[j][i]
                    number = '0'
                elif is_num(temp[j][i]):
                    letter = temp[j][i-1]
                    number = temp[j][i:]
                    break
            letters.append(letter)
            numbers.append(number)
        letters_matrix.append(letters)
        numbers_matrix.append(numbers)
    return letters_matrix, numbers_matrix

def getFamilyName(name):
    '''
    Function which is able to extract the family name from a given series of names
    '''
    family = []
    for test_name in name:
        temp = test_name.split(", ")
        family.append(temp[0])
    return family

def getHonorifics(name):
    '''
    Function which is able to extract the honorifics from a given series of names
    '''
    honorifics = []
    for test_name in name:
        temp = test_name.split(", ")
        hon = temp[-1].split(".")
        honorifics.append(hon[0])
    return honorifics

def preprocessing_model_1(test = False):
    '''
    Function which summarizes the data preprocessing for mlp model 1
    '''
    #Seperating target and dropping cols
    if test == False:
        print('preprocessing the train.csv')
        df_train = pd.read_csv('data/titanic/train.csv')
        df_train_y = df_train['Survived']
        df_train_x = df_train.drop(['PassengerId','Survived','Name','Cabin','Ticket'],axis = 1)
    else:
        print('preprocessing the test.csv')
        df_train = pd.read_csv('data/titanic/test.csv')
        df_train_x = df_train.drop(['PassengerId','Name','Cabin','Ticket'],axis = 1)
    # Treating the missing data in Age and Embarked
    df_train_x.Age[df_train_x.Age.isnull()] = np.float64(30.0)
    df_train_x.Embarked[df_train_x.Embarked.isnull()] = 'S'
    df_train_x.Fare[df_train_x.Fare.isnull()] = 1

    #Classifying Age, SibSp, Parch, Fare
    #Cast the Age into 3 classes <20.125, >20.125 and <38.0, >38.0
    df_train_x.Age[df_train_x.Age <= 20.125] = 0
    df_train_x.Age[(df_train_x.Age <= 38.0) & (df_train_x.Age > 20.125) ] = 1
    df_train_x.Age[df_train_x.Age > 38.0] = 2

    #Cast the Fare into 3 classes: <7.91,7.91< <31.0, >31.0
    df_train_x.Fare[df_train_x.Fare <= 7.91] = 0
    df_train_x.Fare[(df_train_x.Fare > 7.91) & (df_train_x.Fare <=31.0)] = 1
    df_train_x.Fare[df_train_x.Fare > 31.0] = 2

    #Cast the SibSp into 3 classes: 0, 1, >1
    df_train_x.SibSp[df_train_x.SibSp>1] = 2

    #Cast the SibSp into 3 classes: 0, 1, >1
    df_train_x.Parch[df_train_x.Parch>1] = 2

    #one hot encoding
    train_Pclass = (df_train_x.Pclass - 1).as_matrix()
    train_Pclass = keras.utils.to_categorical(train_Pclass, num_classes=3)

    train_Age = (df_train_x.Age).as_matrix()
    train_Age = keras.utils.to_categorical(train_Age, num_classes=3)

    train_SibSp = (df_train_x.SibSp).as_matrix()
    train_SibSp = keras.utils.to_categorical(train_SibSp, num_classes=3)

    train_Parch = (df_train_x.Parch).as_matrix()
    train_Parch = keras.utils.to_categorical(train_Parch, num_classes=3)

    train_Fare = (df_train_x.Fare).as_matrix()
    train_Fare = keras.utils.to_categorical(train_Fare, num_classes=3)

    train_Sex = np.zeros([len(df_train_x.Sex),])
    for i in range(len(df_train_x.Sex)):
        if df_train_x.Sex[i] == 'male':
            train_Sex[i] = 1
    train_Sex = keras.utils.to_categorical(train_Sex, num_classes=2)

    train_Embarked = np.zeros([len(df_train_x.Embarked),])
    for i in range(len(df_train_x.Embarked)):
        if df_train_x.Embarked[i] == 'Q':
            train_Embarked[i] = 1
        elif df_train_x.Embarked[i] == 'S':
            train_Embarked[i] = 2
    train_Embarked = keras.utils.to_categorical(train_Embarked, num_classes=3)
    print("Checking the right shape")
    print('Pclass: ',train_Pclass.shape)
    print('Sex: ',train_Sex.shape)
    print('Age: ',train_Age.shape)
    print('SibSp: ',train_SibSp.shape)
    print('Parch: ',train_Parch.shape)
    print('Fare: ',train_Fare.shape)
    print('Embarked: ',train_Embarked.shape)

    train_x = np.hstack([train_Pclass,train_Sex, train_Age,train_SibSp,train_Parch,train_Fare,train_Embarked])
    if test == False:
        train_y = df_train_y.as_matrix()
        train_y = keras.utils.to_categorical(train_y, num_classes=2)
    else:
        train_y = None
    #Testing
    # print(df_train_x.describe())
    # print(df_train_x.info())
    return train_x, train_y


def preprocessing_model_2(test = False):
    '''
    Function which summarizes the data preprocessing for mlp model 2
    '''
    # Helper functions
    def casting_to_noble(series_honorific,classes):
        '''
        Takes in a list of classes that shall be casted as a Noble class
        '''
        for i in range(len(classes)):
            series_honorific[series_honorific == classes[i]] = "Noble"

    def casting_to_mrs(series_honorific,classes):
        '''
        Takes in a list of classes that shall be casted as a Mrs class
        '''
        for i in range(len(classes)):
            series_honorific[series_honorific == classes[i]] = "Mrs"

    def casting_to_miss(series_honorific,classes):
        '''
        Takes in a list of classes that shall be casted as a Miss class
        '''
        for i in range(len(classes)):
            series_honorific[series_honorific == classes[i]] = "Miss"

    def getLettersAndNumbers(cabin):
        '''
        Function which is able to extract the letters and numbers from a given
        series of cabin names

        outputs 2 matrices of letters and numbers
        '''
        def no_num(x):
            num = ['0','1','2','3','4','5','6','7','8','9']
            if len(x) == 1 and x not in num:
                # print('no number')
                return True
        def is_num(x):
            num = ['0','1','2','3','4','5','6','7','8','9']
            if x in num:
                return True
            return False

        letters_matrix = []
        numbers_matrix = []
        for test in cabin:
            temp = test.split(' ')
            letters = []
            numbers = []
            for j in range(len(temp)):
                if temp[j] == 'Unknown':
                    letter = 'U'
                    number = '0'
                for i in range(len(temp[j])):
                    if no_num(temp[j][i]):
                        letter = temp[j][i]
                        number = '0'
                    elif is_num(temp[j][i]):
                        letter = temp[j][i-1]
                        number = temp[j][i:]
                        break
                letters.append(letter)
                numbers.append(number)
            letters_matrix.append(letters)
            numbers_matrix.append(numbers)
        return letters_matrix, numbers_matrix
    # Helper functions end here
    # Code starts here
    #Get mean age first
    df_train = pd.read_csv('data/titanic/train.csv')
    # Honorific
    honorific = getHonorifics(df_train.Name)
    series_honorific = pd.Series(honorific)
    mean_capt_age = df_train.Age[series_honorific == 'Capt'].mean()
    mean_col_age = df_train.Age[series_honorific == 'Col'].mean()
    mean_don_age = df_train.Age[series_honorific == 'Don'].mean()
    mean_dr_age = df_train.Age[series_honorific == 'Dr'].mean()
    mean_jon_age = df_train.Age[series_honorific == 'Jonkheer'].mean()
    mean_lady_age = df_train.Age[series_honorific == 'Lady'].mean()
    mean_maj_age = df_train.Age[series_honorific == 'Major'].mean()
    mean_master_age = df_train.Age[series_honorific == 'Master'].mean()
    mean_miss_age = df_train.Age[series_honorific == 'Miss'].mean()
    mean_mlle_age = df_train.Age[series_honorific == 'Mlle'].mean()
    mean_mme_age = df_train.Age[series_honorific == 'Mme'].mean()
    mean_mr_age = df_train.Age[series_honorific == 'Mr'].mean()
    mean_mrs_age = df_train.Age[series_honorific == 'Mrs'].mean()
    mean_ms_age = df_train.Age[series_honorific == 'Ms'].mean()
    mean_rev_age = df_train.Age[series_honorific == 'Rev'].mean()
    mean_sir_age = df_train.Age[series_honorific == 'Sir'].mean()
    mean_count_age = df_train.Age[series_honorific == 'the Countess'].mean()
    if test == False:
        print('preprocessing the train.csv')
        y = keras.utils.to_categorical(df_train.Survived.as_matrix(), num_classes=2)

    else:
        print('preprocessing the test.csv')
        df_train = pd.read_csv('data/titanic/test.csv')
        y = None
    # Assigning y to the survived column


    # Feature engineering
    # WCFirst
    series_wcfirst = pd.Series(np.zeros(len(df_train)))
    series_wcfirst[(df_train.Age <= 6) | (df_train.Sex == 'female')] = 1
    df_train["WCFirst"] = series_wcfirst
    # GoodCabin
    df_train.Cabin[df_train.Cabin.isnull()] = "Unknown"
    letters,numbers = getLettersAndNumbers(df_train.Cabin)
    alphabets = []
    for cabin_letters in letters:
        alphabets.append(cabin_letters[0])
    series_alphabet = pd.Series(alphabets)
    series_good_cabin = pd.Series(np.ones(len(df_train)))
    series_good_cabin[(series_alphabet == 'n') | (series_alphabet == 'A')] = 0
    df_train["Cabin_Good"] = series_good_cabin

    # Treating the missing data in Age(**new**) and Embarked
    df_train.Age[(series_honorific == 'Capt') & (df_train.Age.isnull())] = mean_capt_age
    df_train.Age[(series_honorific == 'Col') & (df_train.Age.isnull())] = mean_col_age
    df_train.Age[(series_honorific == 'Don') & (df_train.Age.isnull())] = mean_don_age
    df_train.Age[(series_honorific == 'Dr') & (df_train.Age.isnull())] = mean_dr_age
    df_train.Age[(series_honorific == 'Jonkheer') & (df_train.Age.isnull())] = mean_jon_age
    df_train.Age[(series_honorific == 'Lady') & (df_train.Age.isnull())] = mean_lady_age
    df_train.Age[(series_honorific == 'Major') & (df_train.Age.isnull())] = mean_maj_age
    df_train.Age[(series_honorific == 'Master') & (df_train.Age.isnull())] = mean_master_age
    df_train.Age[(series_honorific == 'Miss') & (df_train.Age.isnull())] = mean_miss_age
    df_train.Age[(series_honorific == 'Mlle') & (df_train.Age.isnull())] = mean_mlle_age
    df_train.Age[(series_honorific == 'Mme') & (df_train.Age.isnull())] = mean_mme_age
    df_train.Age[(series_honorific == 'Mr') & (df_train.Age.isnull())] = mean_mr_age
    df_train.Age[(series_honorific == 'Mrs') & (df_train.Age.isnull())] = mean_mrs_age
    df_train.Age[(series_honorific == 'Ms') & (df_train.Age.isnull())] = mean_ms_age
    df_train.Age[(series_honorific == 'Rev') & (df_train.Age.isnull())] = mean_rev_age
    df_train.Age[(series_honorific == 'Sir') & (df_train.Age.isnull())] = mean_sir_age
    df_train.Age[(series_honorific == 'the Countess') & (df_train.Age.isnull())] = mean_count_age
    # Assigning the missing embarked to S
    df_train.Embarked[df_train.Embarked.isnull()] = 'S'
    print(mean_ms_age)
    print("HERERERERERERERERER")
    # problem =
    print(df_train[df_train.Age.isnull()])
    print("=========================================")
    print(df_train[(series_honorific == 'Ms') & (df_train.Age.isnull())])
    # Casting Honorific into Noble, Mr, Mrs, Miss, Master
    casting_to_noble(series_honorific,["Capt", "Col", "Don", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir", "the Countess"])
    casting_to_mrs(series_honorific, ["Ms","Donna"])
    casting_to_miss(series_honorific, ["Mlle", "Mme"])
    df_train["Honorific"] = series_honorific

    # Dropping columns
    # PassengerId, Survived, Name, Cabin, Ticket, Fare
    if test == False:
        df_train_x = df_train.drop(['PassengerId','Survived','Name','Cabin','Ticket','Fare'],axis = 1)
    else:
        df_train_x = df_train.drop(['PassengerId','Name','Cabin','Ticket','Fare'],axis = 1)
    #Classifying Age, SibSp, Parch
    #Cast the Age into 3 classes <21.773973, >21.773973 and <35.898148, >35.898148
    # print(df_train_x.Age)
    df_train_x.Age[df_train_x.Age <= 21.773973] = 0
    df_train_x.Age[(df_train_x.Age <= 35.898148) & (df_train_x.Age > 21.773973) ] = 1
    df_train_x.Age[df_train_x.Age > 35.898148] = 2
    print(df_train_x.Age.value_counts())
    print(len(df_train_x.Age))
    print(df_train_x[df_train_x.Age.isnull()])
    #Cast the SibSp into 3 classes: 0, 1, >1
    df_train_x.SibSp[df_train_x.SibSp>1] = 2

    #Cast the Parch into 3 classes: 0, 1, >1
    df_train_x.Parch[df_train_x.Parch>1] = 2

    #one hot encoding
    train_Pclass = (df_train_x.Pclass - 1).as_matrix()
    train_Pclass = keras.utils.to_categorical(train_Pclass, num_classes=3)

    train_Age = (df_train_x.Age).as_matrix()
    train_Age = keras.utils.to_categorical(train_Age, num_classes=3)

    train_SibSp = (df_train_x.SibSp).as_matrix()
    train_SibSp = keras.utils.to_categorical(train_SibSp, num_classes=3)

    train_Parch = (df_train_x.Parch).as_matrix()
    train_Parch = keras.utils.to_categorical(train_Parch, num_classes=3)

    train_WCFirst = (df_train_x.WCFirst).as_matrix()
    train_WCFirst = keras.utils.to_categorical(train_WCFirst, num_classes=2)

    train_Cabin_Good = (df_train_x.Cabin_Good).as_matrix()
    train_Cabin_Good = keras.utils.to_categorical(train_Cabin_Good, num_classes=2)

    train_Sex = np.zeros([len(df_train_x.Sex),])
    for i in range(len(df_train_x.Sex)):
        if df_train_x.Sex[i] == 'male':
            train_Sex[i] = 1
    train_Sex = keras.utils.to_categorical(train_Sex, num_classes=2)

    train_Embarked = np.zeros([len(df_train_x.Embarked),])
    for i in range(len(df_train_x.Embarked)):
        if df_train_x.Embarked[i] == 'Q':
            train_Embarked[i] = 1
        elif df_train_x.Embarked[i] == 'S':
            train_Embarked[i] = 2
    train_Embarked = keras.utils.to_categorical(train_Embarked, num_classes=3)

    train_Hon = np.zeros([len(df_train_x.Sex),])
    for i in range(len(df_train_x.Honorific)):
        if df_train_x.Honorific[i] == 'Miss':
            train_Hon[i] = 1
        elif df_train_x.Honorific[i] == 'Mrs':
            train_Hon[i] = 2
        elif df_train_x.Honorific[i] == 'Master':
            train_Hon[i] = 3
        elif df_train_x.Honorific[i] == 'Noble':
            train_Hon[i] = 4
    train_Hon = keras.utils.to_categorical(train_Hon, num_classes=5)

    print("Checking the right shape")
    print('Pclass: ',train_Pclass.shape)
    print('Sex: ',train_Sex.shape)
    print('Age: ',train_Age.shape)
    print('SibSp: ',train_SibSp.shape)
    print('Parch: ',train_Parch.shape)
    print('WCFirst: ',train_WCFirst.shape)
    print('Cabin_Good: ',train_Cabin_Good.shape)
    print('Embarked: ',train_Embarked.shape)
    print('Honorific: ',train_Hon.shape)

    x = np.hstack([train_Pclass,train_Sex, train_Age,train_SibSp,train_Parch,train_WCFirst,train_Cabin_Good,
                   train_Embarked,train_Hon])
    return x,y

def build_mlp_model(dims):
    '''
    build multi-layer perceptron network according to dims

    dims is an array of integers specifiying input, hidden and output layers
    '''
    model = Sequential()
    model.add(Dense(dims[1], activation = 'sigmoid', input_dim = dims[0]))
    for dim in range(1,len(dims) - 2):
        model.add(Dense(dims[dim+1], activation = 'sigmoid'))
        # model.add(Dropout(rate = 0.5))
    model.add(Dense(dims[-1], activation = 'sigmoid'))
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(lr=0.01),
                  metrics=['accuracy'])
    return model

def accuracy_loss_plot(history):
    '''
    Accuracy and loss plot for mlp trained in keras

    history is model.fit

    shows the graphs
    '''
    # list all data in history
    # print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
#     plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
#     # summarize history for loss
    plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
#     plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
# train_x, train_y = preprocessing_model_1()
# print(train_x.shape)
# print(train_y.shape)
