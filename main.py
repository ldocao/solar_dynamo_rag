from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai


PROMPT = "Which articles have been written by Yoshimura ?"


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