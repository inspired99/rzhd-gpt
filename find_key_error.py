import pickle

with open('faiss_pipeline/train_type_to_boarder.pickle', 'rb') as f:
    train_type_to_boarder = pickle.load(f)


print(train_type_to_boarder)
