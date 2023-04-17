import pandas as pd
from utils.all_utils import prepare_data ,save_plot
from utils.model import Perceptron


def main(data ,modelName ,plotName ,eta ,epochs):
    df_OR = pd.DataFrame(data)
    X ,y = prepare_data(df_OR)
    
    # ETA = learning rate
    model = Perceptron(eta = ETA , epochs=EPOCHS)
    model.fit(X,y)

    # _ is just dumy variable
    _ = model.total_loss()


    model.save(filename=modelName ,path='model_or')
    save_plot(df_OR ,model ,filename=plotName)
    

if __name__ == "__main__":

    OR = {
        "x1" :[0,0,1,1],
        "x2" : [0,1,0,1],
        "y" :[0,1 ,1 ,1]
    }
    ETA = 0.3
    EPOCHS = 10
    main(data = OR ,modelName = "or.model", plotName = "or.png" ,eta = ETA ,epochs = EPOCHS)





