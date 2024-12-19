import logging
import uuid
import faiss
import numpy as np
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings

############################################################
# Custom Aggregator Class
############################################################
class CustomMongoAggregator:
    def __init__(self, uri, database, collection, temp_prefix='temp'):
        self.client = MongoClient(uri)
        self.db = self.client[database]
        self.collection = self.db[collection]
        self.custom_operators = {}
        self.temp_prefix = temp_prefix
        self.logger = logging.getLogger(__name__)

    def add_custom_operator(self, name, func):
        if not name.startswith('$'):
            raise ValueError("Custom operator name must start with '$'")
        self.custom_operators[name] = func

    def contains_custom_operator(self, stage):
        def check_expr(expr):
            if isinstance(expr, dict):
                for key, value in expr.items():
                    if key in self.custom_operators:
                        return True
                    elif isinstance(value, (dict, list)):
                        if check_expr(value):
                            return True
            elif isinstance(expr, list):
                for item in expr:
                    if check_expr(item):
                        return True
            return False
        return check_expr(stage)

    def aggregate(self, pipeline):
        current_collection = self.collection
        temp_collections = []

        try:
            pipeline_iter = iter(pipeline)
            sub_pipeline = []
            for stage in pipeline_iter:
                if not self.contains_custom_operator(stage):
                    sub_pipeline.append(stage)
                else:
                    # Execute pipeline without custom operators
                    if sub_pipeline:
                        current_collection = self.execute_sub_pipeline(current_collection, sub_pipeline)
                        sub_pipeline = []
                        temp_collections.append(current_collection)

                    # Process custom stage
                    documents = list(current_collection.find())
                    documents = self.process_custom_stage(documents, stage)

                    # Output to a temp collection
                    temp_collection_name = f"{self.temp_prefix}_{uuid.uuid4().hex}"
                    temp_collection = self.db[temp_collection_name]
                    if documents:
                        temp_collection.insert_many(documents)
                    current_collection = temp_collection
                    temp_collections.append(current_collection)

            # Execute remaining sub_pipeline if any
            if sub_pipeline:
                current_collection = self.execute_sub_pipeline(current_collection, sub_pipeline)
                temp_collections.append(current_collection)

            results = list(current_collection.find())

        except PyMongoError as e:
            self.logger.error(f"MongoDB Error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected Error: {e}")
            raise
        finally:
            # Clean up temporary collections
            for temp_col in temp_collections:
                if temp_col != self.collection:
                    try:
                        temp_col.drop()
                    except Exception as e:
                        self.logger.warning(f"Failed to drop temp collection {temp_col.name}: {e}")

        return results

    def execute_sub_pipeline(self, collection, pipeline):
        temp_collection_name = f"{self.temp_prefix}_{uuid.uuid4().hex}"
        pipeline_with_out = pipeline + [{'$out': temp_collection_name}]
        collection.aggregate(pipeline_with_out)
        return self.db[temp_collection_name]

    def process_custom_stage(self, documents, stage):
        operator, expr = next(iter(stage.items()))
        if operator in ['$project', '$addFields']:
            processed_docs = []
            for doc in documents:
                new_doc = {}
                for key, value in expr.items():
                    if isinstance(value, int) and value == 1:
                        new_doc[key] = doc.get(key)
                    elif value == 0:
                        continue
                    else:
                        new_doc[key] = self.process_expr(value, doc)
                if '_id' in doc:
                    new_doc['_id'] = doc['_id']
                processed_docs.append(new_doc)
            return processed_docs
        else:
            raise NotImplementedError(f"Custom processing for operator {operator} not implemented.")

    def process_expr(self, expr, doc):
        if isinstance(expr, dict):
            if len(expr) == 1:
                key, value = next(iter(expr.items()))
                if key in self.custom_operators:
                    return self.custom_operators[key](doc, value)
                elif key.startswith('$'):
                    return self.evaluate_operator(key, value, doc)
                else:
                    return {key: self.process_expr(value, doc)}
            else:
                return {k: self.process_expr(v, doc) for k, v in expr.items()}
        elif isinstance(expr, list):
            return [self.process_expr(item, doc) for item in expr]
        elif isinstance(expr, str) and expr.startswith('$'):
            return self.get_field_value(doc, expr[1:])
        else:
            return expr

    def evaluate_operator(self, operator, value, doc):
        # No standard operators implemented
        raise NotImplementedError(f"Operator {operator} not implemented.")

    def get_field_value(self, doc, field_path):
        fields = field_path.split('.')
        value = doc
        for f in fields:
            if isinstance(value, dict) and f in value:
                value = value[f]
            else:
                return None
        return value


############################################################
# FAISS Vector Search
############################################################
class VectorSearchIndex:
    def __init__(self, docs, embedding_field):
        self.logger = logging.getLogger(__name__)
        self.docs = docs
        self.embedding_field = embedding_field
        # Using OllamaEmbeddings from langchain_ollama just as an example
        self.model = OllamaEmbeddings(model="nomic-embed-text")
        self.logger.info("Building FAISS index...")
        self.doc_embeddings = self.build_embeddings()
        self.index = self.build_index(self.doc_embeddings)

    def build_embeddings(self):
        texts = [doc.get(self.embedding_field, "") for doc in self.docs]
        embeddings = self.model.embed_documents(texts)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def build_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    def search(self, query_text, top_k=5):
        query_embedding = self.model.embed_documents([query_text])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.docs):
                results.append((self.docs[idx], float(dist)))
        return results

def faiss_search_operator(doc, args, index):
    """
    $faissSearch operator.
    Args: [field_name, top_k, min_score (optional)]
    - field_name: The field from the current doc to use as the query.
    - top_k: How many results to return.
    - min_score: Minimum similarity score to consider.
    """
    logger = logging.getLogger(__name__)

    if len(args) < 2:
        raise ValueError("$faissSearch requires at least field_name and top_k")

    field_name = args[0]
    top_k = int(args[1])
    min_score = float(args[2]) if len(args) > 2 else 0.0

    field_value = doc.get(field_name)
    if field_value is None:
        return []

    search_results = index.search(field_value, top_k=top_k)
    filtered = [(d, s) for (d, s) in search_results if s >= min_score]

    # Return the matched titles and scores
    return [{"title": d[index.embedding_field], "score": s} for d, s in filtered]

############################################################
# DEMO
############################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Connect to MongoDB and insert demo data
    mongo_db = CustomMongoAggregator(
        uri="mongodb://localhost:27017/?directConnection=true",
        database="mydatabase",
        collection="mycollection"
    )
    mongo_db.collection.delete_many({})

    # Insert documents with distinct movie titles and ratings
    DATASET = [
        {"_id": 1, "title": "A Journey Through Time", "rating": 5, "status": "active"},
        {"_id": 2, "title": "Adventures in Space", "rating": 5, "status": "active"},
        {"_id": 3, "title": "Midnight Mystery", "rating": 4, "status": "active"},
        {"_id": 4, "title": "Time Traveler's Delight", "rating": 5, "status": "active"},
        {"_id": 5, "title": "The Great Outdoors", "rating": 3, "status": "active"},
        {"_id": 6, "title": "Exploring Mars", "rating": 5, "status": "active"},
        {"_id": 7, "title": "Space Adventures Continued", "rating": 4, "status": "active"},
        {"_id": 8, "title": "Back to the Future", "rating": 5, "status": "active"},
        {"_id": 9, "title": "A Night at the Museum", "rating": 4, "status": "active"},
        {"_id": 10,"title": "Journey into the Unknown", "rating": 5, "status": "active"}
    ]

    mongo_db.collection.insert_many(DATASET)

    # Build vector index on 'title' field
    all_docs = list(mongo_db.collection.find())
    vs_index = VectorSearchIndex(all_docs, embedding_field='title')

    def faiss_search_wrapper(doc, args):
        return faiss_search_operator(doc, args, vs_index)

    # Add faiss search operator
    mongo_db.add_custom_operator('$faissSearch', faiss_search_wrapper)

    # Aggregation pipeline:
    # 1) Match docs with rating > 3 and active
    # 2) Perform FAISS search for similar titles (top 5, min_score 0.8 for high relevance)
    pipeline = [
        {
            '$match': {
                'rating': {'$gt': 3},
                'status': 'active'
            }
        },
        {
            '$project': {
                'title': 1,
                'similar_titles': {
                    '$faissSearch': ['title', 5, 0.8]
                }
            }
        }
    ]

    try:
        output = mongo_db.aggregate(pipeline)
    except Exception as e:
        logging.error(f"Aggregation failed: {e}")
        output = []

    # Print results
    for doc in output:
        print(f"\nDocument ID {doc.get('_id')}:")
        print(f"Title: {doc.get('title')}")
        print("Similar Titles:")
        for st in doc.get('similar_titles', []):
            print(f" - {st['title']} (score: {st['score']:.2f})")
