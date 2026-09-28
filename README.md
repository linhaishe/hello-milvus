# Docker 部署 Milvus CRUD Demo

当前代码使用的是本地模型不需要外部 AI API.

流程 ：文本 → 本地 embedding 模型 → 向量 → Milvus

```
运行 demo
  ↓
删除 Milvus 中的 knowledge_docs 集合 (如果有)
  ↓
读取 knowledge_db
  ↓
本地模型生成向量
  ↓
SDK 写入 Milvus ( knowledge_docs ) 服务
  ↓
Milvus 保存到 Docker 挂载目录 ./volumes/
```

`./volumes/` 是 Milvus 的持久化数据目录，不是向量文件的简单文本形式，而是 Milvus 管理的内部存储格式。

## 1. 启动 Milvus

需要先安装 Docker 和 Docker Compose：

```bash
docker compose up -d
docker compose ps
```

## 2. 安装 Python SDK

建议使用虚拟环境：

```bash
python3 -m venv .venv # 建一个独立的 Python 环境  venv - py 内置模块
source .venv/bin/activate # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## 3. 运行真实知识库 CRUD 示例

```bash
python demo.py
```

脚本会演示：

- 读取 `knowledge_db/` 下的 Markdown、TXT、SRT、VTT、TSV、JSON 文件
- 将长文本切分后，用 `paraphrase-multilingual-MiniLM-L12-v2` 生成向量
- 将文本块和向量插入 Milvus
- 按条件查询
- 根据问题进行向量相似度搜索
- 使用 `upsert` 更新记录（Milvus 没有单独的 update API）
- 按条件删除记录

首次运行会自动下载 embedding 模型。PDF 和 MP4 当前会跳过；如需处理它们，需要先增加 PDF 文本提取或视频转录步骤。

停止服务：

```bash
docker compose down
```

如需同时删除本地 Milvus 数据，再手动删除 `volumes/` 目录。

# QA

##  `text = path.read_text(encoding="utf-8", errors="ignore")`

比如读取这个文件：

```text
knowledge_db/prompt_engineering/1. 简介 Introduction.md
```

代码：

```python
text = path.read_text(
    encoding="utf-8",
    errors="ignore"
)
```

此时 `text` 是一个 Python 字符串，类型是：

```python
<class 'str'>
```

内容大概是：

```python
'# 第一章 简介\n\n欢迎来到**面向开发者的提示工程**部分，本部分内容基于**吴恩达老师的《Prompt Engineering for Developer》课程**进行编写。《Prompt Engineering for Developer》课程是由**吴恩达老师**与 OpenAI 技术团队成员 **Isa Fulford** 老师合作授课。'
```

注意：

```python
\n
```

表示换行。

如果打印出来：

```python
print(text)
```

看到的是：

```text
# 第一章 简介

欢迎来到**面向开发者的提示工程**部分，本部分内容基于**吴恩达老师的《Prompt Engineering for Developer》课程**进行编写。《Prompt Engineering for Developer》课程是由吴恩达老师与 OpenAI 技术团队成员 Isa Fulford 老师合作授课。
```

所以它不是字典，也不是列表，而是普通字符串：

```python
text = "文件里的全部文本内容"
```

后面代码：

```python
text = " ".join(text.split())
```

会把它整理成一行：

```python
'# 第一章 简介 欢迎来到**面向开发者的提示工程**部分，本部分内容基于...'
```

然后再把这段字符串切成多个文本块，交给 embedding 模型生成向量。

## 获取到文件内容里有`\n\n`这个，在什么时候处理的

会处理，但不是在 `read_text()` 这一步处理。

`read_text()` 只是原样读取文件，所以内容中仍然有：

```python
"\n\n"
```

后面这句才会处理：

```python
text = " ".join(text.split())
```

例如：

```python
text = "第一段\n\n第二段\n\n第三段"
```

执行后：

```python
text = "第一段 第二段 第三段"
```

它会：

- 把换行转换为空格
- 把多个连续空白压缩成一个空格
- 去掉开头和结尾的空白

因此：

```text
第一段


第二段
```

会变成：

```text
第一段 第二段
```

这样做是为了让文本更适合切分和生成向量。

不过这也会丢失“段落边界”。如果希望保留段落信息，可以改成只清理多余空格，不完全去掉换行。对于当前 demo，简单压成一行足够运行。

## `for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):`

这段代码的作用是：**把长文本切成多个有重叠的文本块**。

```python
for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
    chunk = text[start : start + CHUNK_SIZE]

    if chunk:
        documents.append({
            "source": str(path),
            "text": chunk
        })
```

当前配置：

```python
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
```

所以每次移动：

```python
800 - 100 = 700
```

假设文本长度是 2000：

```text
第 1 块：text[0:800]
第 2 块：text[700:1500]
第 3 块：text[1400:2000]
```

可以看到：

```text
第 1 块和第 2 块重叠 100 个字符
第 2 块和第 3 块重叠 100 个字符
```

重叠的目的，是避免一句话刚好被切在两个文本块之间，导致上下文丢失。

### `chunk = ...`

```python
chunk = text[start : start + CHUNK_SIZE]
```

这是字符串切片：

```python
text[开始位置:结束位置]
```

例如：

```python
text[0:800]
```

表示取第 0 到第 799 个字符。

### `if chunk`

```python
if chunk:
```

表示只在文本块不为空时保存。

### `documents.append`

最终 `documents` 大概是一个列表：

```python
[
    {
        "source": "knowledge_db/example.md",
        "text": "第一个文本块..."
    },
    {
        "source": "knowledge_db/example.md",
        "text": "第二个文本块..."
    }
]
```

后面每个字典会被转换成一条 Milvus 数据。

## 接入rag

```text
用户问题
  ↓
生成问题向量
  ↓
Milvus 搜索相关文本
  ↓
拼接文本上下文
  ↓
发送给大模型生成答案
```

当前代码已经完成前 3 步：

```python
question_vector = model.encode(
    [question],
    normalize_embeddings=True
)[0].tolist()

results = client.search(
    COLLECTION,
    data=[question_vector],
    limit=3,
    output_fields=["source", "text"]
)
```

RAG 还需要取出搜索结果：

```python
contexts = [
    hit["entity"]["text"]
    for hit in results[0]
]
context = "\n\n".join(contexts)
```

然后组成提示词：

```python
prompt = f"""
请根据以下资料回答问题。

资料：
{context}

问题：
{question}
"""
```

最后把 `prompt` 发送给 OpenAI、通义、DeepSeek 等大模型 API。

所以：

```text
Milvus：负责找资料
大模型：负责根据资料回答
run_demo：负责串起整个流程
```

如果只是做本地检索，不需要接入大模型；接入大模型后才是完整 RAG。






