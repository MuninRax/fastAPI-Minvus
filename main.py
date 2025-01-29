from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import pandas as pd
import langid
import config 

# FastAPI instance
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load configurations from config.py
MILVUS_HOST = config.MILVUS_HOST
MILVUS_PORT = config.MILVUS_PORT
COLLECTION_NAME = config.COLLECTION_NAME
EN_MODEL = config.EN_MODEL_NAME
TH_MODEL = config.TH_MODEL_NAME
METRIC_TYPE = config.METRIC_TYPE
DIMENSION = config.DIMENSION
N_LIST = config.N_LIST
N_PROBE = config.N_PROBE
TOP_K = config.TOP_K
DATA_FILE = "flattened_price_list.json"

# Initialize SentenceTransformer model
en_model = SentenceTransformer(EN_MODEL)
th_model = SentenceTransformer(TH_MODEL)


# Load data as DataFrame
data = pd.read_json(DATA_FILE, encoding='utf-8')
data.fillna('', inplace=True)
data.insert(0, 'id', range(1, len(data) + 1))
data['price'] = data['price'].astype(str)
columns_to_combine = ['description', 'category_name' ,'note']
data['combined'] = data[columns_to_combine].apply(lambda row: ' '.join(row.astype(str)), axis=1)


def create_collection_and_insert_data():
    """
    Create the Milvus collection and insert data if not already initialized.
    """
    try:
        # # Connect to Milvus
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

        # Check if the collection exists
        if not utility.has_collection(COLLECTION_NAME):
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="product_code", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="category_name", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="note", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="combined", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="embedding_english", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),  
                FieldSchema(name="embedding_thai", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),    
            ]
            schema = CollectionSchema(fields, description="Dual-embedding schema for product search")

            # Create collection
            collection = Collection(name=COLLECTION_NAME, schema=schema)
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": METRIC_TYPE,
                "params": {"nlist": N_LIST},
            }
            collection.create_index(field_name="embedding_english", index_params=index_params)
            collection.create_index(field_name="embedding_thai", index_params=index_params)

            
            # Add embeddings to the data
            data['embedding_english'] = data['combined'].apply(lambda x: en_model.encode(x).tolist())
            data['embedding_thai'] = data['combined'].apply(lambda x: th_model.encode(x).tolist())
            
            # Insert data
            insert_data = [
                data['id'].tolist(),
                data['product_code'].tolist(),
                data['description'].tolist(),
                data['category_name'].tolist(),
                data['note'].tolist(),
                data['combined'].tolist(),
                data['embedding_english'].tolist(),
                data['embedding_thai'].tolist(),
            ]
            collection.insert(insert_data)
            collection.flush()
            collection.load()
            print("Collection created and data inserted successfully.")
            return {"status": "success", "message": "Collection created and data inserted successfully."}
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists.")
            return {"status": "exists", "message": f"Collection '{COLLECTION_NAME}' already exists."}
    except Exception as e:
        print(f"Error during collection creation or data insertion: {e}")
        return {"status": "error", "message": str(e)}


# create collection on start up
@app.on_event("startup")
def startup_event():
    create_collection_and_insert_data()


# Query request model
class QueryRequest(BaseModel):
    query_text: str
    limit: int = TOP_K


# Semantic search
@app.post("/price-list-search", tags=['price list search'])
def search_products(request: QueryRequest):
    try:
        query = request.query_text
        query_vector = None

        # Detect query language
        lang = langid.classify(query)[0]

        # Choose the model and embedding based on the language
        if lang == 'en':  # English
            query_vector = en_model.encode(query).tolist()
            embedding_field = "embedding_english"
        else:  # Assume Thai or other languages
            query_vector = th_model.encode(query).tolist()
            embedding_field = "embedding_thai"

        # Search in Milvus
        collection = Collection(name=COLLECTION_NAME)
        search_params = {
            "metric_type": METRIC_TYPE,
            "params": {"nprobe": N_PROBE},
        }
        results = collection.search(
            data=[query_vector],
            anns_field=embedding_field,
            param=search_params,
            limit=request.limit,
            output_fields=["product_code", "description", "category_name", "note"],
        )

        retrieved_ids = [result.id for result in results[0]]
        filtered_data = data[data['id'].isin(retrieved_ids)]
        filtered_data['id'] = pd.Categorical(filtered_data['id'], categories=retrieved_ids, ordered=True)
        filtered_data = filtered_data.sort_values('id')

        output_results = []
        for _, row in filtered_data.iterrows():
            output_results.append({
                "no": row["no"],
                "product_code": row["product_code"],
                "description": row["description"],
                "unit": row["unit"],
                "price": row["price"],
                "category_name": row["category_name"],
                "note": row["note"]
            })

        return {"results": output_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Add UI
@app.get("/", response_class=HTMLResponse)
def render_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
