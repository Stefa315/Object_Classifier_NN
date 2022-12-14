import os
from keras.models import model_from_json


def Save_New_Model_Multi(model,model_mode):
    # serialize model to JSON for models with DropOut and BatchNOrmalization
    model_version = len( os.listdir("../Models_Saved/{}".format(model_mode)))
    os.makedirs("../Models_Saved/{}/model_{}".format(model_mode,model_version),exist_ok=True)
    model_json = model.to_json()
    with open("../Models_Saved/{}/model_{}/model.json".format(model_mode,model_version),"w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../Models_Saved/{}/model_{}/model_weights.h5".format(model_mode,model_version),overwrite=True)
    print("Saved model to disk")

def Save_New_Model(model,model_mode):
    # serialize model to JSON for models with DropOut and BatchNOrmalization
    model_version = len( os.listdir("../Models_Saved/{}".format(model_mode)))+1
    os.makedirs("../Models_Saved/{}/model_{}".format(model_mode,model_version),exist_ok=True)
    model_json = model.to_json()
    with open("../Models_Saved/{}/model_{}/model.json".format(model_mode,model_version),"w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../Models_Saved/{}/model_{}/model_weights.h5".format(model_mode,model_version),overwrite=True)
    print("Saved model to disk")

def Load_Latest_Model(model_number=None,model_mode=None):

    if model_number==None:
        model_version = len(os.listdir("../Models_Saved/{}".format(model_mode)))
        print("MODEL VERSION", model_version)
        # load json and create model
        json_file = open('../Models_Saved/{}/model_{}/model.json'.format(model_mode,model_version), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("../Models_Saved/{}/model_{}/model_weights.h5".format(model_mode,model_version))
        # print("Loaded model from disk")
    else:
        # load json and create model
        json_file = open('../Models_Saved/{}/model_{}/model.json'.format(model_mode,model_number), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("../Models_saved/{}/model_{}/model_weights.h5".format(model_mode,model_number))
        # print("Loaded model from disk")

    return loaded_model