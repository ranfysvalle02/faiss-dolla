import logging
import uuid
import faiss
import numpy as np
from pymongo import MongoClient
from pymongo.errors import PyMongoError
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
                    if sub_pipeline:
                        current_collection = self.execute_sub_pipeline(current_collection, sub_pipeline)
                        sub_pipeline = []
                        temp_collections.append(current_collection)
                    documents = list(current_collection.find())
                    documents = self.process_custom_stage(documents, stage)
                    temp_collection_name = f"{self.temp_prefix}_{uuid.uuid4().hex}"
                    temp_collection = self.db[temp_collection_name]
                    if documents:
                        temp_collection.insert_many(documents)
                    current_collection = temp_collection
                    temp_collections.append(current_collection)
            if sub_pipeline:
                current_collection = self.execute_sub_pipeline(current_collection, sub_pipeline)
                temp_collections.append(current_collection)
            results = list(current_collection.find())
        except PyMongoError as e:
            self.logger.error(f"MongoDB Error: {e}")
            raise
        except Exception as e:
            self.logger.exception("Unexpected Error:")
            raise
        finally:
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
    def __init__(self, docs, embedding_fields):
        self.logger = logging.getLogger(__name__)
        self.docs = docs
        self.embedding_fields = embedding_fields if isinstance(embedding_fields, list) else [embedding_fields]
        self.model = OllamaEmbeddings(model="nomic-embed-text")
        self.logger.info("Building FAISS index...")
        self.doc_embeddings = self.build_embeddings()
        if len(self.doc_embeddings) == 0:
            raise ValueError("No embeddings were generated. Check your fields and data.")
        self.index = self.build_index(self.doc_embeddings)

    def build_embeddings(self):
        aggregated_embeddings = []
        for doc in self.docs:
            field_texts = []
            for field in self.embedding_fields:
                field_value = doc.get(field, "")
                if isinstance(field_value, list):
                    field_texts.extend([str(item) for item in field_value if isinstance(item, str)])
                elif isinstance(field_value, str):
                    field_texts.append(field_value)
            if not field_texts:
                continue
            try:
                embeddings = self.model.embed_documents(field_texts)
                embeddings = np.array(embeddings)
                if embeddings.ndim != 2:
                    continue
                avg_embedding = np.mean(embeddings, axis=0)
                norm_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                aggregated_embeddings.append(norm_embedding)
            except:
                continue
        if not aggregated_embeddings:
            return np.array([])
        return np.vstack(aggregated_embeddings).astype('float32')

    def build_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    def search(self, query_texts, top_k=5):
        if not query_texts:
            return []
        try:
            query_embeddings = self.model.embed_documents(query_texts)
            query_embeddings = np.array(query_embeddings).astype('float32')
            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)
            faiss.normalize_L2(query_embeddings)
            distances, indices = self.index.search(query_embeddings, top_k)
            results = []
            for idx, dist in zip(indices, distances):
                for i, d in zip(idx, dist):
                    if i < len(self.docs):
                        results.append((self.docs[i], float(d)))
            return results
        except:
            return []

def faiss_search_operator(doc, args, index):
    if len(args) < 2:
        raise ValueError("$faissSearch requires field_name and top_k")
    field_name = args[0]
    top_k = int(args[1])
    min_score = float(args[2]) if len(args) > 2 else 0.0
    field_value = doc.get(field_name)
    if field_value is None:
        return []
    if isinstance(field_value, list):
        query_texts = [str(item) for item in field_value if isinstance(item, str)]
        if not query_texts:
            return []
        search_results = index.search(query_texts, top_k=top_k)
    elif isinstance(field_value, str):
        search_results = index.search([field_value], top_k=top_k)
    else:
        return []
    filtered = [(d, s) for (d, s) in search_results if s >= min_score]
    return [{"text": d['description'], "score": s} for d, s in filtered]

############################################################
# DEMO
############################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mongo_db = CustomMongoAggregator(
        uri="mongodb://localhost:27017/?directConnection=true",
        database="mydatabase",
        collection="products"
    )
    mongo_db.collection.delete_many({})

    DATASET = [
        {"_id": 1, "description": "High-quality wireless headset", "categories": ["Electronics", "Audio"], "rating": 5, "status": "active"},
        {"_id": 2, "description": "Ergonomic office chair with lumbar support", "categories": ["Furniture", "Office"], "rating": 5, "status": "active"},
        {"_id": 3, "description": "Durable stainless steel water bottle", "categories": ["Kitchen", "Outdoor"], "rating": 4, "status": "active"},
        {"_id": 4, "description": "Bluetooth speaker with long battery life", "categories": ["Electronics", "Audio"], "rating": 5, "status": "active"},
        {"_id": 5, "description": "Comfortable cotton t-shirt", "categories": ["Apparel"], "rating": 3, "status": "active"},
        {"_id": 6, "description": "Noise-cancelling over-ear headphones", "categories": ["Electronics", "Audio"], "rating": 5, "status": "active"},
        {"_id": 7, "description": "Adjustable standing desk converter", "categories": ["Furniture", "Office"], "rating": 4, "status": "active"},
        {"_id": 8, "description": "Compact portable charger", "categories": ["Electronics", "Gadgets"], "rating": 5, "status": "active"},
        {"_id": 9, "description": "Ceramic coffee mug", "categories": ["Kitchen"], "rating": 4, "status": "active"},
        {"_id": 10,"description": "Lightweight running shoes", "categories": ["Apparel", "Outdoor"], "rating": 5, "status": "active"},
        {"_id": 11, "description": "Insulated lunch box", "categories": ["Kitchen", "Outdoor"], "rating": 4, "status": "active"}
    ]

    mongo_db.collection.insert_many(DATASET)

    all_docs = list(mongo_db.collection.find())
    vs_index = VectorSearchIndex(all_docs, embedding_fields=['description', 'categories'])

    def faiss_search_wrapper(doc, args):
        return faiss_search_operator(doc, args, vs_index)

    mongo_db.add_custom_operator('$faissSearch', faiss_search_wrapper)

    pipeline1 = [
        {
            '$match': {
                'rating': {'$gt': 3},
                'status': 'active'
            }
        },
        {
            '$project': {
                'description': 1,
                'similar_products': {
                    '$faissSearch': ['description', 5, 0.8]
                }
            }
        }
    ]

    pipeline2 = [
        {
            '$match': {
                'rating': {'$gte': 4},
                'status': 'active'
            }
        },
        {
            '$project': {
                'description': 1,
                'categories': 1,
                'similar_category_products': {
                    '$faissSearch': ['categories', 3, 0.7]
                }
            }
        }
    ]

    try:
        output1 = mongo_db.aggregate(pipeline1)
    except Exception as e:
        logger.error(f"Aggregation Pipeline 1 failed: {e}")
        output1 = []

    try:
        output2 = mongo_db.aggregate(pipeline2)
    except Exception as e:
        logger.error(f"Aggregation Pipeline 2 failed: {e}")
        output2 = []

    print("\n--- Pipeline 1: Similar Products by Description ---")
    for doc in output1:
        print(f"\nID: {doc.get('_id')}")
        print(f"Description: {doc.get('description')}")
        print("Similar Products:")
        for sp in doc.get('similar_products', []):
            print(f" - {sp['text']} (score: {sp['score']:.2f})")

    print("\n--- Pipeline 2: Similar Products by Category ---")
    for doc in output2:
        print(f"\nID: {doc.get('_id')}")
        print(f"Description: {doc.get('description')}")
        print(f"Categories: {', '.join(doc.get('categories', []))}")
        print("Similar Products:")
        for sp in doc.get('similar_category_products', []):
            print(f" - {sp['text']} (score: {sp['score']:.2f})")
