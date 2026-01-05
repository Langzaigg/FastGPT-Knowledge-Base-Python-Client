"""
FastGPT 文件上传示例 - 支持本地文件和URL链接
"""
from fastgpt import FastGPTKnowledgeBase

# 配置信息
API_KEY = "your-api-key-here"
BASE_URL = "http://your-base-url/api"

# 初始化客户端
kb = FastGPTKnowledgeBase(BASE_URL, API_KEY)

# 示例：创建知识库
dataset_id = "your-dataset-id"

# ============================================================
# 示例 1: 从本地文件上传（支持的格式：pdf、xlsx、pptx、docx、md、txt）
# ============================================================
try:
    print("上传本地文件...")
    collection_id = kb.create_file_collection(
        dataset_id=dataset_id,
        file_path="./example.pdf",  # 本地文件路径
        training_type="chunk"
    )
    print(f"✓ 成功创建集合: {collection_id}")
except ValueError as e:
    print(f"✗ 文件格式错误: {e}")
except FileNotFoundError as e:
    print(f"✗ 文件未找到: {e}")
except Exception as e:
    print(f"✗ 上传失败: {e}")

# ============================================================
# 示例 2: 从URL链接上传
# ============================================================
try:
    print("\n从URL下载并上传文件...")
    collection_id = kb.create_file_collection(
        dataset_id=dataset_id,
        file_path="https://example.com/document.pdf",  # URL链接
        training_type="chunk",
        chunk_setting_mode="custom",
        chunk_size=1000,
        index_size=512
    )
    print(f"✓ 成功创建集合: {collection_id}")
except ValueError as e:
    print(f"✗ 文件格式错误: {e}")
except Exception as e:
    print(f"✗ 上传失败: {e}")

# ============================================================
# 示例 3: 上传不支持的格式会报错
# ============================================================
try:
    print("\n尝试上传不支持的格式...")
    collection_id = kb.create_file_collection(
        dataset_id=dataset_id,
        file_path="https://example.com/image.jpg",  # 不支持的格式
        training_type="chunk"
    )
except ValueError as e:
    print(f"✓ 正确拦截不支持的格式: {e}")
except Exception as e:
    print(f"✗ 其他错误: {e}")

# ============================================================
# 支持的文件格式列表
# ============================================================
print("\n支持的文件格式：")
print("- PDF (.pdf)")
print("- Excel (.xlsx)")
print("- PowerPoint (.pptx)")
print("- Word (.docx)")
print("- Markdown (.md)")
print("- 文本文件 (.txt)")
