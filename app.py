from flask import Flask,request,render_template
from models import word2vec_model,query_embedding,recipe_embeddings,get_top_ingredients,recipes,plot_top_ingredients,CHART_PATH,create_pie_chart,PIE_CHART_PATH
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("base.html")



@app.route('/find_result', methods=['GET', 'POST'])
def find(): 
    search_query = request.args.get('search')  
    print(search_query)
    query_embed = query_embedding(search_query, word2vec_model)  
    query_sim = cosine_similarity([query_embed], recipe_embeddings)  
    top_recipes = query_sim.argsort()[0][::-1]  
    keys = recipes.iloc[top_recipes][:8]  
    print(keys)
    return render_template('index.html', search_query=keys)  


@app.route('/trending')
def trending():
    top_ingredients = get_top_ingredients(recipes, top_n=10)
    plot_top_ingredients(top_ingredients, CHART_PATH)
    return render_template('chart.html', chart_path=CHART_PATH)


@app.route('/nutrition', methods=['GET'])
def nutrition():
    nutrition_sums = recipes[['FatContent', 'SaturatedFatContent', 'CholesterolContent', 
                              'SodiumContent', 'CarbohydrateContent', 'FiberContent', 
                              'SugarContent', 'ProteinContent']].sum()

    create_pie_chart(nutrition_sums, PIE_CHART_PATH)

    return render_template('nutrition.html', chart_path=PIE_CHART_PATH)

if __name__=="__main__":
    app.run(debug=True)