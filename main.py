from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import pandas as pd
import config 

# FastAPI instance
app = FastAPI()

# Load configurations from config.py
MILVUS_HOST = config.MILVUS_HOST
MILVUS_PORT = config.MILVUS_PORT
COLLECTION_NAME = config.COLLECTION_NAME
EMBED_NAME = config.EMBED_NAME
METRIC_TYPE = config.METRIC_TYPE
DIMENSION = config.DIMENSION
N_LIST = config.N_LIST
N_PROBE = config.N_PROBE
TOP_K = config.TOP_K
DATA_FILE = "flattened_price_list.json"

# Initialize SentenceTransformer model
model = SentenceTransformer(EMBED_NAME)

# Load DataFrame
data = pd.read_json(DATA_FILE)
data.fillna('', inplace=True)
data.insert(0, 'id', range(1, len(data) + 1))
data['combined'] = data.drop(columns=['id']).apply(lambda row: ' '.join(row.astype(str)), axis=1)


def create_collection_and_insert_data():
    """
    Create the Milvus collection and insert data if not already initialized.
    """
    try:
        # Connect to Milvus
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

        # Check if the collection exists
        if not utility.has_collection(COLLECTION_NAME):
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="combined", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
            ]
            schema = CollectionSchema(fields, description="Collection for product data")

            # Create collection
            collection = Collection(name=COLLECTION_NAME, schema=schema)
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": METRIC_TYPE,
                "params": {"nlist": N_LIST},
            }
            collection.create_index(field_name="embedding", index_params=index_params)

            # Add embeddings to the data
            data['embedding'] = data['combined'].apply(lambda x: model.encode(x).tolist())

            # Insert data
            insert_data = [
                data['id'].tolist(),
                data['combined'].tolist(),
                data['embedding'].tolist(),
            ]
            collection.insert(insert_data)
            collection.load()
            print("Collection created and data inserted successfully.")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception as e:
        print(f"Error during collection creation or data insertion: {e}")


# FastAPI Startup Event
@app.on_event("startup")
def startup_event():
    # Create collection and insert data on app startup
    create_collection_and_insert_data()


# Query request model
class QueryRequest(BaseModel):
    query_text: str
    limit: int = TOP_K


# Endpoint for searching products
@app.post("/search")
def search_products(request: QueryRequest):
    try:
        # Encode query text
        query_vector = model.encode(request.query_text).tolist()
        collection = Collection(name=COLLECTION_NAME)

        # Search parameters
        search_params = {
            "metric_type": METRIC_TYPE,
            "params": {"nprobe": N_PROBE},
        }
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=request.limit,
            output_fields=["id", "combined"],
        )

        # Retrieve and sort results
        retrieved_ids = [result.id for result in results[0]]
        filtered_data = data[data['id'].isin(retrieved_ids)]
        filtered_data['id'] = pd.Categorical(filtered_data['id'], categories=retrieved_ids, ordered=True)
        filtered_data = filtered_data.sort_values('id')

        # Select only the required columns
        filtered_data = filtered_data[
            ['no', 'product_code', 'description', 'unit', 'price', 'category_name', 'note']
        ]

        return {"results": filtered_data.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
