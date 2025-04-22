"""
- très très rapide de faire un premier POC 
- très sensible au fait qu'il ne puisse pas parser tous les documents (j'avais un docx et RTF), pas de message d'erreur, il faut commencer avec seulement quelques documents dont on est sur que ca fonctionne
- il est capable de parser des types de pdf très différents (images et texte), car j'avais des fichiers qui datent de 1990
- possibilité de choisir l'embedding, la distance treshold, chunk size et overlap
- tres peu de doc, par exemple je voulais changer le type de distance pour faire le retrieval (euclidean) mais j'ai pas trouvé. Meme si de base, le cosine distance est celui qu'il faut utiliser avec les embedding preconstruits
- algorithmes de recherche dans les vector database:
    - a priori, il faudrait calculer le produit entre le vecteur input et tous les vecteurs de la base
    - locally sensitive hashing
    - partitionnement spatial ou pre calcul avec du K means clustering
    - product quantization
    - approximate nearest neighbors: hierarchical navigable small world (HNSW)

- possibilité de brancher le vector database que tu veux entre pinecone et weaviate
- dans l'exemple de base RAG ManagedDB, on ne voit la database nulle part, vraiment juste pour faire du POC
- je dois reinitialiser le vector database a chaque fois avec le RAG ManagedDB, d'où le fait que ce ne soit pas production ready
"""

from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai


PROMPT = "What is an alpha dynamo? Give me an extended answer of at least 200 words with equations"


PROJECT_ID = "long-456911"
display_name = "soho"
paths = ["https://drive.google.com/drive/folders/1AsrrQbAhEQrNx5s3mmArR6mpEgsNOWpz"]  # requires giving access the driver folder to the service account vertex AI agent

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

# Create RagCorpus
# Configure embedding model, for example "text-embedding-005".
DEFAULT_EMBEDDING = "publishers/google/models/text-embedding-005"
embedding_model_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model=DEFAULT_EMBEDDING
    )
)

rag_corpus = rag.create_corpus(
    display_name=display_name,
    backend_config=rag.RagVectorDbConfig(
        rag_embedding_model_config=embedding_model_config
    ),
)

# Import Files to the RagCorpus
response = rag.import_files(
    rag_corpus.name,
    paths,
    # Optional
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(
            chunk_size=512,
            chunk_overlap=100,
        ),
    ),
    max_embedding_requests_per_min=1000,  # Optional
)
print(f"Imported {response.imported_rag_files_count} files.")

# Direct context retrieval
rag_retrieval_config = rag.RagRetrievalConfig(
    top_k=5,  # Optional
    filter=rag.Filter(vector_distance_threshold=0.5),  # Optional
)
response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(
            rag_corpus=rag_corpus.name,
            # Optional: supply IDs from `rag.list_files()`.
            # rag_file_ids=["rag-file-1", "rag-file-2", ...],
        )
    ],
    text=PROMPT,
    rag_retrieval_config=rag_retrieval_config,
)
print(response)


# Enhance generation
# Create a RAG retrieval tool
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=rag_corpus.name,  # Currently only 1 corpus is allowed.
                    # Optional: supply IDs from `rag.list_files()`.
                    # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                )
            ],
            rag_retrieval_config=rag_retrieval_config,
        ),
    )
)

# Create a Gemini model instance
rag_model = GenerativeModel(
    model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool]
)

# Generate response
response = rag_model.generate_content(PROMPT)
print(response.text)