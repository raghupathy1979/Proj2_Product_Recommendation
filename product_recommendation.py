import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample products with some features
products = pd.DataFrame({
    'productId': ['P1', 'P2', 'P3', 'P4', 'P5'],
    'productName': ['Red Shirt', 'Blue Jeans', 'Green Shirt', 'Black Shoes', 'White Hat'],
    'category': ['Clothing', 'Clothing', 'Clothing', 'Footwear', 'Accessories'],
    'tags': ['shirt red casual', 'jeans blue denim', 'shirt green casual', 'shoes black leather', 'hat white summer']
})

products.set_index('productId', inplace=True)

# Combine category and tags into a single string for vectorization
products['text_features'] = products['category'] + " " + products['tags']

# Vectorize text features
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(products['text_features'])

# Compute pairwise cosine similarity between products
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a reverse map of productId to index
indices = pd.Series(products.index)

def recommend_products(product_id, top_n=3):
    idx = indices[indices == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # exclude the item itself

    recommended_ids = [products.index[i[0]] for i in sim_scores]
    return products.loc[recommended_ids][['productName']]

# Example: Recommend similar products to "P1" (Red Shirt)
recommendations = recommend_products('P1')
print("Recommendations for Red Shirt:")
print(recommendations)