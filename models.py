from gensim.models import Word2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ast
import numpy as np 
from collections import Counter
import matplotlib.pyplot as plt
import os

CHART_PATH = os.path.join('static', 'top_ingredients_chart.png')
PIE_CHART_PATH = os.path.join('static', 'pie_chart.png')

word2vec_model = Word2Vec.load('word2vec_cbow.model')
recipes = pd.read_csv('clean_df.csv')
recipes['IngredientList'] = recipes['IngredientList'].apply(ast.literal_eval)




def get_recipe_embedding(ingredient_list, model):
    vectors = [word2vec_model.wv[ingredient] for ingredient in ingredient_list if ingredient in word2vec_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)  # Average the vectors
    else:
        return np.zeros(model.vector_size)
    
recipe_embeddings = np.array([get_recipe_embedding(ingredients, word2vec_model) for ingredients in recipes['IngredientList'][:25000]])


def query_embedding(query, model):
    words = query.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
    
def get_top_ingredients(recipes, top_n=10):
    
    ingredient_counts = Counter(
        [ingredient for ingredients in recipes['IngredientList'] for ingredient in ingredients]
    )
    return ingredient_counts.most_common(top_n)


def plot_top_ingredients(top_ingredients, save_path):
    
    ingredients, counts = zip(*top_ingredients)  # Unpack ingredients and their counts
    plt.figure(figsize=(10, 6))
    plt.bar(ingredients, counts, color="skyblue", edgecolor="black", linewidth=1.2)
    plt.title("Top Ingredients in Recipes", fontsize=16, fontweight='bold')
    plt.xlabel("Ingredients", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)  # Save the chart as an image
    plt.close()

def create_pie_chart(data, save_path):
    labels = data.index
    values = data.values

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Nutritional Content Distribution', fontsize=16, fontweight='bold')
    plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()