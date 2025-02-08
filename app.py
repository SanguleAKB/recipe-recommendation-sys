from flask import Flask,request,render_template,jsonify
from models import word2vec_model,query_embedding,recipe_embeddings,get_top_ingredients,recipes,plot_top_ingredients,CHART_PATH,create_pie_chart,PIE_CHART_PATH
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import tensorflow as tf
app = Flask(__name__)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@app.route('/')
def index():
    return render_template("base.html")

ann_model = tf.keras.models.load_model('my_model.keras')

@app.route('/find_result', methods=['GET', 'POST'])
def find(): 
    search_query = request.args.get('search')  
    query_embed = query_embedding(search_query, word2vec_model)  
    query_sim = cosine_similarity([query_embed], recipe_embeddings)  
    top_recipes = query_sim.argsort()[0][::-1]  
    keys = recipes.iloc[top_recipes][:8]  
    return render_template('index.html', search_query=keys)  


@app.route('/trending')
def trending():
    top_ingredients = get_top_ingredients(recipes, top_n=10)
    plot_top_ingredients(top_ingredients, CHART_PATH)
    return render_template('chart.html', chart_path=CHART_PATH)

@app.route('/nutrition', methods=['GET'])
def nutrition():
    index = request.args.get('index')
    # Validate the index
    if index is None:
        return "Index is missing", 400
    try:
        index = int(index)
    except ValueError:
        return "Invalid index provided", 400

    try:
        nutrition_data = recipes.loc[index, [
            'FatContent', 'SaturatedFatContent', 'CholesterolContent',
            'SodiumContent', 'CarbohydrateContent', 'FiberContent',
            'SugarContent', 'ProteinContent'
        ]]
    except KeyError:
        return "Recipe index out of range", 404

    # Ensure data is a Pandas Series
    if isinstance(nutrition_data, pd.Series):
        create_pie_chart(nutrition_data, PIE_CHART_PATH)
    else:
        return "Error: Nutritional data is not in the expected format", 500

    key = recipes.loc[index]
    # Render the nutrition template
    return render_template('nutrition.html', chart_path=PIE_CHART_PATH,recipe=key)

@app.route('/predict',methods=['post','get'])
def predict(): 
    cal = request.form.get('calories')
    fat = request.form.get('fat')
    carbohydrates = request.form.get('carbohydrate')
    protein = request.form.get('protein')
    cholesterol = request.form.get('cholesterol')
    sodium = request.form.get('sodium')
    fiber = request.form.get('fiber')
    sugar = request.form.get('sugar')

    input_data = np.array([cal, fat, carbohydrates, protein,cholesterol,sodium,fiber,sugar], dtype=float).reshape(1, -1)

    # Make prediction
    prediction = ann_model.predict(input_data)
    predicted_category = np.argmax(prediction, axis=1)[0]

    key = recipes[recipes['RecipeCategoryEncode']==int(predicted_category)]
    shuffled_key = key.sample(frac=1).reset_index(drop=True)[:20]
    # Render result
    return render_template('recommended.html',category=shuffled_key)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=50000, threaded=True)
