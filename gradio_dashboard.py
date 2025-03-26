import pandas as pd
import numpy as np

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd

import gradio as gr
import os

load_dotenv()

movies = pd.read_csv("modified_dataset_with_index.csv")

movies["large_thumbnail"] = movies["Poster_Link"] + "&fife=w800"

movies["large_thumbnail"] = (
                                movies['Poster_Link'] 
                                + "&quality=100"       # Max quality
                                + "&auto=webp"         # Modern format
                                + "&width=2000"        # Higher resolution
                            )

# movies["large_thumbnail"] = np.where(movies["large_thumbnail"].isna(),"cover_not_found.jpg",movies["large_thumbnail"])

if os.path.exists("description_embeddings"):

    db_movies = Chroma(persist_directory="description_embeddings", embedding_function=OpenAIEmbeddings())

else:

    loader = TextLoader("tagged_description.txt", encoding="utf-8")

    raw_documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")

    documents = text_splitter.split_documents(raw_documents)

    db_movies = Chroma.from_documents(documents,embedding=OpenAIEmbeddings(),persist_directory="description_embeddings")



def retrieve_semantic_recommendations(query: str, certificate: str, year_ranges:str, initial_top_k: int = 20, final_top_k : int = 30 ) -> pd.DataFrame:
    # Get semantic search results
    recs = db_movies.similarity_search(query, k=initial_top_k)
    
    movies_list = []
    similarity_scores = []  
    
    for rec in recs:
        # Extract movie ID and similarity score
        content = rec.page_content.strip('"')

        movies_list.append(int(content.split(" ")[0]))

        similarity_scores.append(rec.metadata.get('score', 0))  # Get similarity score
    
    # Get matching movies from dataset
    retrieved_set = movies[movies["index"].isin(movies_list)].copy()
    
    # Add similarity scores and combine with Hybrid_Score
    retrieved_set['Similarity_Score'] = retrieved_set['index'].map(
        dict(zip(movies_list, similarity_scores))
    )
    
    # Normalize scores to same scale (0-100) if needed
    retrieved_set['Similarity_Score'] = retrieved_set['Similarity_Score'] * 100
    
    # Combine scores (adjust weights as needed)
    retrieved_set['Combined_Score'] = (
        0.5 * retrieved_set['Hybrid_Score'] + 
        0.5 * retrieved_set['Similarity_Score']
    )

    # Sort by combined score
    retrieved_set = retrieved_set.sort_values('Combined_Score', ascending=False)

    #  if the category passed is not all
    if certificate != "ALL":
        # retrieve recommendations to final top k
        retrieved_set = retrieved_set[retrieved_set["Certificate_Group"] == certificate][:final_top_k]

    else:
        
        retrieved_set = retrieved_set.head(final_top_k)

    # if the year range passed is not all
    if year_ranges != "ALL":

        retrieved_set = retrieved_set[retrieved_set["Year-range"] == year_ranges][:final_top_k]

    else:

        retrieved_set = retrieved_set.head(final_top_k)
    

    return retrieved_set


def recommend_movies(
    query:str,
    category:str,
    year_ranges:str
    ):
    
    recommendations = retrieve_semantic_recommendations(query, category,year_ranges)

    results = []

    for _,row in recommendations.iterrows():

        description = row["Overview"]

        
        actors = row["Star1"] + "," + row["Star2"] + "," + row["Star3"]

        director = row["Director"]


        caption = f"{row['Series_Title']} Starring {actors} directed by {director} : {description}"

        results.append((row["large_thumbnail"], caption))


    return results



categories = ["ALL"] + sorted(movies["Certificate_Group"].unique())

year_ranges = ["ALL"] + sorted(movies["Year-range"].unique())


with gr.Blocks(theme = gr.themes.Glass()) as dashboard:

    gr.Markdown("# Semantic Movie Recommender")

    with gr.Row():

        user_query = gr.Textbox(label = "Please enter your movie or what you like to watch",
                                placeholder = "eg. Spiderman")
        # default value all
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a Category", value = "ALL")

        year_range_dropdown = gr.Dropdown(choices = year_ranges, label = "Select a Year Range", value = "ALL")

        # tone_dropdown = gr.Dropdown(choices = tones,label = "Select a Emotional Tone", value = "ALL")

        submit_button = gr.Button("Find Recommendation")

    gr.Markdown("## Recommendations")

    output = gr.Gallery(label = "Recommend books", columns=8, rows =2)

    submit_button.click(fn = recommend_movies, inputs = [user_query, category_dropdown, year_range_dropdown],
                        outputs = output)



    if __name__ == "__main__":

            dashboard.launch()




        