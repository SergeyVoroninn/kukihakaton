from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class embedding:

    def __init__(self):
        model_path = "models"  # ← путь к папке, где лежат файлы модели

        self.model = SentenceTransformer(model_path)

    def get_embeddings(self, text):
        """
        Return embedding vector for given text
        """
        embeddings = self.model.encode_query(text)
        return embeddings
    
    def get_similarity(self,emb1,emb2):
        """
        Returns a similarity matrix of given embeddings.
        """
        similarity_matrix = cosine_similarity(emb1,emb2)
        # Выводит вектор схожести (список)
        output = []
        for row in similarity_matrix:
            max = 0
            for el in row:
                if el > max:
                    max = el
            output.append(max)        

        return output