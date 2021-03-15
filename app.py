import csv
from flask import Flask, render_template, request
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


class Books:
    def __init__(self):
        self.books = pd.read_csv('Book/Books.csv')
        self.users = pd.read_csv('Book/Users.csv')
        self.ratings = pd.read_csv('Book/Ratings.csv')
        # Fractionnement des évaluations d'utilisateurs explicites et implicites
        # nous supprimons l'ensemble de notation qui a la note de 0
        self.ratings_explicit = self.ratings[self.ratings.bookRating != 0]
        self.ratings_implicit = self.ratings[self.ratings.bookRating == 0]

        # Nombre de notes moyennes de chaque livre et nombre total de notes
        self.average_rating = pd.DataFrame(
            self.ratings_explicit.groupby('ISBN')['bookRating'].mean())
        self.average_rating['ratingCount'] = pd.DataFrame(
            self.ratings_explicit.groupby('ISBN')['bookRating'].count())
        self.average_rating = self.average_rating.rename(
            columns={'bookRating': 'MeanRating'})
        # Pour obtenir des similitudes plus fortes
        counts1 = self.ratings_explicit['userID'].value_counts()
        self.ratings_explicit = self.ratings_explicit[
            self.ratings_explicit['userID'].isin(counts1[counts1 >= 50].index)]

        # Explicit Books and ISBN
        self.explicit_ISBN = self.ratings_explicit.ISBN.unique()
        self.explicit_books = self.books.loc[self.books['ISBN'].isin(
            self.explicit_ISBN)]
        # Livres explicites et ISBN
        # Rechercher un dict pour Book et BookID
        self.Book_lookup = dict(
            zip(self.explicit_books["ISBN"], self.explicit_books["bookTitle"]))
        self.ID_lookup = dict(
            zip(self.explicit_books["bookTitle"], self.explicit_books["ISBN"]))

    def Top_Books(self, n=10, RatingCount=100, MeanRating=3):
        # ici nous spécifions la latence de meanRating avec une valeur de 3
        # et latence de RatingCount avec une valeur de 100
        # cela constitue une valeur seuil pour prédire les meilleurs ensembles de livres possibles pour l'utilisateur
        # livres avec la note la plus élevée
        # cette fonction ne recommandera aucun livre affiche uniquement les livres les mieux notés par chaque utilisateur
        BOOKS = self.books.merge(self.average_rating, how='right', on='ISBN')
        M_Rating = BOOKS.loc[BOOKS.ratingCount >= RatingCount].sort_values(
            'MeanRating', ascending=False).head(n)

        H_Rating = BOOKS.loc[BOOKS.MeanRating >= MeanRating].sort_values(
            'ratingCount', ascending=False).head(n)

        return M_Rating, H_Rating


class KNN(Books):

    def __init__(self, n_neighbors=5):
        # calling super class __init__ method
        super().__init__()
        # assigning k  value = 5
        self.n_neighbors = n_neighbors
        # removing nan value
        self.ratings_mat = self.ratings_explicit.pivot(
            index="ISBN", columns="userID", values="bookRating").fillna(0)
        '''
        Implémentation de kNN
        Dans l'analyse numérique et le calcul scientifique, une matrice ou un tableau épars est une matrice dans laquelle
        la plupart des éléments sont nuls.
        Nous convertissons notre table en une matrice 2D et remplissons les valeurs manquantes avec des zéros
        (puisque nous allons calculer les distances entre les vecteurs de notation). Nous transformons ensuite les valeurs (notes)
        de la matrice de données dans une matrice scipy clairsemée pour des calculs plus efficaces.
        Recherche des voisins les plus proches Nous utilisons des algorithmes non supervisés avec sklearn.neighbours.
        L'algorithme que nous utilisons pour calculer les voisins les plus proches est «brute», et nous spécifions «métrique = cosinus»
        L'algorithme calculera la similitude cosinus entre les vecteurs de notation. Enfin, nous adaptons le modèle.
        '''
        self.uti_mat = csr_matrix(self.ratings_mat.values)
        # Raccord de modèle KNN
        # Raccord de modèle KNN
        # en utilisant la similitude cosinus
        '''Mathématiquement, il mesure
        le cosinus de l'angle entre deux vecteurs projetés dans un espace multidimensionnel
        La similarité cosinus est une métrique utilisée pour déterminer comment
         similaires, les documents sont quelle que soit leur taille.'''
        self.model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model_knn.fit(self.uti_mat)

    def Recommend_Books(self, book, n_neighbors=10):
        bID = self.ID_lookup[book]
        query_index = self.ratings_mat.index.get_loc(bID)
        KN = self.ratings_mat.iloc[query_index, :].values.reshape(1, -1)
        distances, indices = self.model_knn.kneighbors(
            KN, n_neighbors=n_neighbors + 1)
        Rec_books = list()
        Book_dis = list()

        for i in range(1, len(distances.flatten())):
            Rec_books.append(self.ratings_mat.index[indices.flatten()[i]])
            Book_dis.append(distances.flatten()[i])

        Book = self.Book_lookup[bID]

        Recommmended_Books = self.books[self.books['ISBN'].isin(Rec_books)]


        return Book, Recommmended_Books, Book_dis


# -----------------------------------------------------------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    global KNN_Recommended_Books
    if request.method == 'POST':
        ICF = KNN()
        book = request.form['book']
        data = book
        try:

            _, KNN_Recommended_Books, _ = ICF.Recommend_Books(data)

            KNN_Recommended_Books = KNN_Recommended_Books.merge(
                ICF.average_rating, how='left', on='ISBN')
            KNN_Recommended_Books = KNN_Recommended_Books.rename(
                columns={'bookRating': 'MeanRating'})

            df = pd.DataFrame(KNN_Recommended_Books, columns=['bookTitle', 'bookAuthor', 'MeanRating'])
            products_list = df.values.tolist()
            return render_template('result.html',products_list=products_list,book_value=book)
        except:
            return render_template("error.html")


if __name__ == '__main__':
    app.run(debug=True)
