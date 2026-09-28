import hashlib  # 根据文本生成唯一 ID
import json
from pathlib import Path

from pymilvus import DataType, MilvusClient
from sentence_transformers import SentenceTransformer

COLLECTION = "knowledge_docs"
KNOWLEDGE_DIR = Path("knowledge_db")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SUPPORTED_SUFFIXES = {".md", ".txt", ".srt", ".vtt", ".tsv", ".json"}
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def read_documents(root=KNOWLEDGE_DIR):
    documents = []
    # 找到目录下所有文件和文件夹，并排序 / rglob("*"): 递归查找所有内容，包括子目录,r 表示 recursive
    # sorted: 按路径名称排序，让每次读取文件的顺序保持一致
    for path in sorted(Path(root).rglob("*")):
        # path.is_file(): 只处理文件，不处理文件夹
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        # 以 UTF-8 编码读取文件内容,如果遇到无法解析的字符，就跳过这些字符，不让程序报错。
        if path.suffix.lower() == ".json":
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False)
                # ensure_ascii=False: 保留中文字符，不转换成 ASCII 编码形式。把 JSON 重新转成文本，同时保留中文，方便后续生成向量
            except json.JSONDecodeError:
                pass
        text = " ".join(text.split())
        # 把长文本切成多个有重叠的文本块
        for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[start : start + CHUNK_SIZE]
            if chunk:  # 表示只在文本块不为空时保存
                documents.append({"source": str(path), "text": chunk})
    return documents


def make_id(source, text):
    """
    根据文件路径和文本内容生成一个数字 ID

    文件路径 + 文本内容
        ↓
    SHA-1 哈希字符串
            ↓
    取前 15 位
            ↓
    转换成整数
            ↓
    得到 Milvus 的 id
    """
    return int(hashlib.sha1(f"{source}\n{text}".encode()).hexdigest()[:15], 16)


def run_demo(uri="http://localhost:19530"):
    documents = read_documents()
    if not documents:
        raise RuntimeError(f"{KNOWLEDGE_DIR} 中没有可读取的文本文件")

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        [item["text"] for item in documents], normalize_embeddings=True
    )
    dimension = len(vectors[0]) # 获取第一个向量的长度，也就是向量维度

    client = MilvusClient(uri=uri)
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)

    #  enable_dynamic_field: 不允许插入 schema 中没有定义的字段
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("source", DataType.VARCHAR, max_length=1000)
    schema.add_field("text", DataType.VARCHAR, max_length=2000)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)

    # 创建一个“索引配置对象”
    # field_name="vector"：给 vector 字段建立索引
    # index_type="AUTOINDEX"：让 Milvus 自动选择索引方式
    # metric_type="COSINE"：使用余弦相似度衡量向量距离
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
    )
    # 创建 collection 时把索引配置一起交给 Milvus
    # Milvus 的 create_collection() 会同时处理 schema 和向量索引配置
    """
    类似于
    CREATE TABLE knowledge_docs (
    id BIGINT PRIMARY KEY,
    source VARCHAR,
    text VARCHAR,
    vector VECTOR
    );
    """
    client.create_collection(COLLECTION, schema=schema, index_params=index_params)

    rows = [
        {
            "id": make_id(item["source"], item["text"]),
            "source": item["source"],
            "text": item["text"],
            "vector": vector.tolist(),
        }
        for item, vector in zip(documents, vectors)
    ]
    """
    zip(documents, vectors):
    把两个列表中相同位置的元素配对
    zip(documents, vectors)
    documents = ["文本A", "文本B", "文本C"]
    vectors = ["向量A", "向量B", "向量C"]
    list(zip(documents, vectors))
    [
    ("文本A", "向量A"),
    ("文本B", "向量B"),
    ("文本C", "向量C")
    ]
    """
    
    client.insert(COLLECTION, rows)
    
    print(f"插入 {len(rows)} 个文本块")
    print(
        "查询:",
        client.query(
            COLLECTION,
            filter="id >= 0",
            limit=3, # 最多返回 3 条结果
            output_fields=["id", "source", "text"],
        ),
    )

    question = "强化学习和提示词工程"
    """
    Milvus 只能比较向量，不能直接理解字符串,所以encode成向量
    model.encode(["问题1", "问题2"])
    [
    [0.12, 0.03, -0.45, ...]
    ]
    
    normalize_embeddings=True：
    导入资料时用了，搜索问题时也要保持一致，余弦相似度比较才一致
    
    tolist():
    把 NumPy 数组转换成普通 Python 列表。
    模型返回的向量通常是 NumPy 数组: array([0.12, 0.03, -0.45])
    vector.tolist() = [0.12, 0.03, -0.45]
    
    """
    question_vector = model.encode([question], normalize_embeddings=True)[0].tolist()
    print(
        "向量搜索:",
        client.search(
            COLLECTION,
            data=[question_vector],
            limit=3,
            output_fields=["source", "text"],
        ),
    )

    first = rows[0]
    updated = {**first, "text": first["text"] + " [已更新]"}
    client.upsert(COLLECTION, [updated])
    print(
        "更新:",
        client.query(
            COLLECTION, filter=f"id == {first['id']}", output_fields=["source", "text"]
        ),
    )

    client.delete(COLLECTION, filter=f"id == {first['id']}")
    print(
        "删除后:",
        client.query(
            COLLECTION, filter=f"id == {first['id']}", output_fields=["id", "source"]
        ),
    )
    client.close()


if __name__ == "__main__":
    run_demo()
