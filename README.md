# FastGPT Knowledge Base Python Client

这是一个轻量级的 Python 客户端封装，用于与 FastGPT 知识库 API 进行交互。它提供了创建知识库、导入数据（文本、链接、文件）、以及进行搜索测试的便捷方法。

特别优化了文件上传功能，支持本地文件路径和远程 URL 直接上传，并自动处理中文文件名的编码问题。

## ✨ 主要功能

* **知识库管理**：创建、获取详情、删除知识库。
* **多模式数据导入**：
* **文件上传**：支持本地文件和网络 URL 图片/文档上传。
* **文本导入**：直接导入纯文本数据。
* **链接导入**：导入网页链接进行抓取。


* **智能处理**：
* 自动识别 URL 与本地路径。
* 自动处理 `multipart/form-data` 的中文文件名编码。
* 支持自定义分块（Chunking）配置。


* **类型提示**：使用 Python `dataclass` 提供清晰的响应数据模型（Data Models），便于开发。

## 🛠️ 依赖安装

本项目依赖 `requests` 库。

```bash
pip install requests

```

## 🚀 快速开始

### 1. 初始化客户端

将 `fastgpt.py` 放入您的项目目录中，然后按照以下方式初始化：

```python
from fastgpt import FastGPTKnowledgeBase

# 配置 API 信息
API_KEY = "fastgpt-xxxxxx"
BASE_URL = "https://api.fastgpt.in/api"  # 或您的私有部署地址

# 初始化
kb = FastGPTKnowledgeBase(base_url=BASE_URL, api_key=API_KEY)

```

### 2. 上传文件（本地或 URL）

这是最常用的功能。`create_file_collection` 方法会自动判断传入的是本地路径还是 URL。

**支持的文件格式**：PDF, Excel (.xlsx), PowerPoint (.pptx), Word (.docx), Markdown (.md), Text (.txt)。

```python
dataset_id = "your-dataset-id"

# --- 方式 A: 上传本地文件 ---
try:
    collection_id = kb.create_file_collection(
        dataset_id=dataset_id,
        file_path="./documents/manual.pdf",
        training_type="chunk",  # 或 'qa'
        chunk_size=1000,
        chunk_split_mode="custom"
    )
    print(f"本地文件上传成功，集合ID: {collection_id}")
except Exception as e:
    print(f"上传失败: {e}")

# --- 方式 B: 从 URL 上传文件 ---
try:
    collection_id = kb.create_file_collection(
        dataset_id=dataset_id,
        file_path="https://example.com/reports/2023_summary.pdf",
        name="2023年度报告.pdf"  # 可选：重命名文件
    )
    print(f"URL文件上传成功，集合ID: {collection_id}")
except Exception as e:
    print(f"上传失败: {e}")

```

### 3. 搜索测试

```python
results = kb.search_test(
    dataset_id=dataset_id,
    text="如何重置密码？",
    limit=3,
    similarity=0.7
)

for item in results:
    print(f"分数: {item.score} | 内容: {item.a[:50]}...")

```

## 📚 API 参考

### `FastGPTKnowledgeBase` 类

#### 知识库操作

* `create_dataset(...)`: 创建新的知识库应用。
* `get_dataset_detail(dataset_id)`: 获取知识库的详细配置信息。
* `delete_dataset(dataset_id)`: 删除知识库。

#### 数据集合（Collection）操作

* **`create_file_collection(...)`**: 核心方法。
* `file_path`: 本地绝对/相对路径，或 HTTP/HTTPS 链接。
* `training_type`: 训练模式，`chunk` (分块) 或 `qa` (问答拆分)。
* `chunk_setting_mode`: `auto` (自动) 或 `custom` (自定义)。
* `custom_pdf_parse`: 是否使用自定义 PDF 解析器。


* `create_text_collection(...)`: 导入纯文本。
* `create_link_collection(...)`: 导入网页链接。
* `push_data(...)`: 手动推送 QA 问答对数据。

## ⚠️ 注意事项

1. **文件格式限制**：仅支持 `.pdf`, `.xlsx`, `.pptx`, `.docx`, `.md`, `.txt`。上传不支持的格式会抛出 `ValueError`。
2. **中文文件名**：代码内部已处理 `urllib.parse.quote` 编码，无需在调用前手动编码文件名。
3. **URL 上传原理**：当使用 URL 上传时，客户端会自动下载文件到临时目录，上传完毕后自动清理。

## 示例代码

查看 [example_file_upload.py](example_file_upload.py) 获取完整的错误处理和文件上传演示代码。
