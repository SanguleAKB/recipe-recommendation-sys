# Recipe Recommendation System 🍔 🍕

## Project Overview
  ### Problem Statement
  In our daily lives, we often have ingredients at home but struggle to decide what to cook with them. Additionally, finding recipes that meet specific nutritional 
  requirements (e.g., high protein, low fat) can be challenging. This project aims to solve these problems by providing personalized recipe recommendations based 
  on available ingredients and nutritional preferences.

  ### Solution
  - This project is a recipe recommendation system that helps users:

  #### Find Recipes Based on Ingredients:
  - Users can input the ingredients they have, and the system will recommend recipes that can be made using those ingredients.
  - Uses Word2Vec CBOW (Continuous Bag of Words) to generate ingredient embeddings and cosine similarity to find the most relevant recipes.

 #### Find Recipes Based on Nutritional Requirements:
  - Users can specify nutritional preferences (e.g., high protein, low sodium), and the system will recommend recipes that meet those requirements.
  - Uses an Artificial Neural Network (ANN) to filter and rank recipes based on nutritional content.

## Database 
- I used Food.com kaggle dataset Data with over 500,000 recipes and 1,400,000 reviews from Food.com. Visit this kaggle link for more details.

## Key Features
  ### Ingredient-Based Search:
  - Recommends recipes based on the ingredients available at home.
  - Uses Word2Vec CBOW to understand the context and relationships between ingredients.
  - Uses cosine similarity to find the most similar recipes.

### Nutrition-Based Recommendations:
- Recommends recipes based on user-defined nutritional requirements (e.g., protein, fat, sodium).
- Uses an ANN to analyze and rank recipes based on nutritional content.

### User-Friendly Interface:
- Built with Flask for the backend and a simple web interface for easy interaction.

## Technologies Used
  ### Natural Language Processing (NLP):
  - Word2Vec CBOW: To generate embeddings for ingredients.
  - Cosine Similarity: To compute similarity between ingredient lists.

  ### Machine Learning:
  - Artificial Neural Network (ANN): To filter and rank recipes based on nutritional requirements.

## Web Framework:
  - Flask: To create a web-based interface for user interaction.

## Deployement using Docker
  ### Why Docker?
By using Docker, you can ensure that the environment in which the application is exactly the same as the environment in which it was built, which can help prevent unexpected issues and improve model performance. Additionally, Docker allows for easy scaling and management of the deployment, making it a great choice for larger machine learning projects.

  ### Docker-Compose
My project is composed of different services (frontend,API). Therefore, our application should run on multiple containers. With the help of Docker-compose we can share our application using the yaml file that define the services that runs together.

### Project Architecture


## Data Processing:
- Pandas: For data manipulation and analysis.
- NumPy: For numerical computations.

## How It Works
- Ingredient-Based Recommendations:
- Users input the ingredients they have.
- The system converts the ingredients into embeddings using Word2Vec CBOW.
- It computes cosine similarity between the user's ingredients and all recipes to find the best matches.

## Nutrition-Based Recommendations:
- Users specify their nutritional requirements (e.g., high protein, low fat).
- The system uses an ANN to analyze the nutritional content of recipes and recommend the best matches.

