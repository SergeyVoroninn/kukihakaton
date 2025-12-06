import embedding
import os
if __name__ == "__main__":

    titles = [
    "iPhone 13 128GB черный",
    "Apple iPhone 13 128 ГБ, чёрный",
    "Samsung Galaxy S21 128GB"
    ] 
    titles2 = [
    " Galaxy jopa",
    "iPhone dick",
    "Не груша wef",
    "Samsung Galaxy S21 128GB"
    ]  
    emb = embedding.embedding()
    test_emb_data1 = emb.get_embeddings(titles)
    print(test_emb_data1)
    test_emb_data2 = emb.get_embeddings(titles2)
    sim_matrix = emb.get_similarity(test_emb_data1,test_emb_data2)
    # print("Матрица схожести:")
    # print(sim_matrix)