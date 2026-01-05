import requests
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union, Type, TypeVar
from urllib.parse import quote, urlparse, unquote

# ==========================================
# 1. 响应数据模型封装 (Data Models)
# ==========================================

T = TypeVar('T')

@dataclass
class BaseResponse:
    """API 基础响应结构"""
    code: int           # 状态码 (200 为成功)
    message: str        # 响应消息/错误提示
    statusText: str     # 状态文本
    data: Any = None    # 实际数据载荷

    @property
    def is_success(self) -> bool:
        return 200 <= self.code < 300

@dataclass
class ModelConfig:
    """模型配置信息"""
    model: str                  # 模型具体名称 (如 text-embedding-ada-002)
    name: str                   # 模型展示名称
    charsPointsPrice: float     # 价格 (每千 token 或积分)
    defaultToken: int           # 默认最大 Token 数

@dataclass
class DatasetDetail:
    """知识库详情对象"""
    id: str                     # 知识库 ID (_id)
    parentId: Optional[str]     # 父级 ID (文件夹模式用)
    type: str                   # 类型 (dataset / folder)
    name: str                   # 知识库名称
    avatar: str                 # 头像 URL
    intro: str                  # 简介
    status: str                 # 状态 (active 等)
    permission: str             # 权限 (private / public)
    vectorModel: ModelConfig    # 向量模型配置
    agentModel: ModelConfig     # 对话模型配置
    canWrite: bool              # 当前用户是否可写
    isOwner: bool               # 当前用户是否是拥有者

    @classmethod
    def from_dict(cls, data: Dict) -> 'DatasetDetail':
        """从字典解析为对象"""
        return cls(
            id=data.get('_id', ''),
            parentId=data.get('parentId'),
            type=data.get('type', 'dataset'),
            name=data.get('name', ''),
            avatar=data.get('avatar', ''),
            intro=data.get('intro', ''),
            status=data.get('status', ''),
            permission=data.get('permission', ''),
            vectorModel=ModelConfig(**data.get('vectorModel', {})),
            agentModel=ModelConfig(**data.get('agentModel', {})),
            canWrite=data.get('canWrite', False),
            isOwner=data.get('isOwner', False)
        )

@dataclass
class PushResults:
    """数据导入结果统计"""
    insertLen: int              # 成功插入的数据条数
    overToken: List[Any] = field(default_factory=list)  # 超出 Token 限制的数据 (未入库)
    repeat: List[Any] = field(default_factory=list)     # 重复的数据 (未入库)
    error: List[Any] = field(default_factory=list)      # 处理发生错误的数据

@dataclass
class CollectionCreateResult:
    """创建集合 (文本/链接/文件) 的返回结果"""
    collectionId: str           # 创建成功的集合 ID
    insertLen: int = 0          # 插入的块数量

@dataclass
class SearchResultItem:
    """单条搜索结果"""
    id: str                     # 数据 ID
    q: str                      # 问题 / 索引内容
    a: str                      # 答案 / 详细内容
    datasetId: str              # 所属知识库 ID
    collectionId: str           # 所属集合 ID
    sourceName: str             # 来源名称 (文件名/网页标题)
    sourceId: Optional[str]     # 来源 ID
    score: float                # 匹配分数 (相似度)

@dataclass
class SearchTestResult:
    """搜索测试返回列表"""
    list: List[SearchResultItem]

    @classmethod
    def from_list(cls, data_list: List[Dict]) -> 'SearchTestResult':
        items = [
            SearchResultItem(
                id=item.get('id', ''),
                q=item.get('q', ''),
                a=item.get('a', ''),
                datasetId=item.get('datasetId', ''),
                collectionId=item.get('collectionId', ''),
                sourceName=item.get('sourceName', ''),
                sourceId=item.get('sourceId'),
                score=item.get('score', 0.0)
            ) for item in data_list
        ]
        return cls(list=items)

# ==========================================
# 2. 核心客户端类 (Client)
# ==========================================

class FastGPTKnowledgeBase:
    def __init__(self, base_url: str, api_key: str):
        """
        初始化客户端
        :param base_url: API 基础地址 (如 http://localhost:3000/api)
        :param api_key: FastGPT API Key
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _post(self, endpoint: str, json_data: Dict = None, **kwargs) -> Dict:
        """内部 POST 请求"""
        return self._request("POST", endpoint, json=json_data, **kwargs)

    def _get(self, endpoint: str, params: Dict = None) -> Dict:
        """内部 GET 请求"""
        return self._request("GET", endpoint, params=params)
    
    def _delete(self, endpoint: str, params: Dict = None) -> Dict:
        """内部 DELETE 请求"""
        return self._request("DELETE", endpoint, params=params)

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()
            res_json = response.json()
            
            # 基础错误拦截
            if res_json.get('code') and res_json['code'] not in [200, 201]:
                raise Exception(f"FastGPT API Error: {res_json.get('message')}")
            
            return res_json
        except Exception as e:
            raise Exception(f"Request Failed: {str(e)}")

    # ---------------------- 知识库管理 ----------------------

    def create_dataset(
        self, 
        name: str, 
        intro: str = "", 
        type: str = "dataset",
        avatar: str = "/icon/logo.svg", 
        vector_model: str = "qwen3-embedding-8b",
        agent_model: str = "qwen3-next-80b-a3b-instruct",
        vlm_model: str = "qwen3-vl-30b-a3b-instruct",
        parent_id: Optional[str] = None
    ) -> str:
        """
        创建知识库
        :return: dataset id
        """
        payload = {
            "parentId": parent_id,
            "type": type,
            "name": name,
            "intro": intro,
            "avatar": avatar,
            "vectorModel": vector_model,
            "agentModel": agent_model,
            "vlmModel": vlm_model
        }
        res = self._post("/core/dataset/create", json_data=payload)
        return res['data']

    def get_dataset_detail(self, dataset_id: str) -> DatasetDetail:
        """
        获取知识库详情
        :return: DatasetDetail 对象
        """
        res = self._get("/core/dataset/detail", params={"id": dataset_id})
        return DatasetDetail.from_dict(res['data'])

    def delete_dataset(self, dataset_id: str) -> bool:
        """
        删除知识库
        :return: 是否成功
        """
        self._delete("/core/dataset/delete", params={"id": dataset_id})
        return True

    # ---------------------- 集合(数据源)管理 ----------------------

    def create_empty_collection(
        self,
        dataset_id: str,
        name: str,
        type: Optional[str] = "folder",
        parent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        创建一个空的集合
        :return: 集合 ID (collectionId)
        """
        payload = {
            "datasetId": dataset_id,
            "name": name,
            "parentId": parent_id,
            "type": type,
            "metadata": metadata or {}
        }
        if tags:
            payload["tags"] = tags
        
        res = self._post("/core/dataset/collection/create", json_data=payload)
        # API 返回的 data 直接是 collectionId 字符串
        return res['data']

    def create_text_collection(
        self,
        text: str,
        dataset_id: str,
        name: str,
        training_type: str = "chunk",  # chunk 或 qa
        parent_id: Optional[str] = None,
        index_prefix_title: Optional[bool] = None,
        custom_pdf_parse: bool = False,
        auto_indexes: Optional[bool] = None,
        image_index: Optional[bool] = None,
        chunk_setting_mode: str = "auto",  # auto 或 custom
        chunk_split_mode: str = "size",  # size 或 char
        chunk_size: int = 1500,
        index_size: int = 512,
        chunk_splitter: str = "",
        qa_prompt: str = "",
        tags: Optional[List[str]] = None,
        create_time: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        导入文本数据
        :return: 集合 ID (collectionId)
        """
        payload = {
            "text": text,
            "datasetId": dataset_id,
            "name": name,
            "parentId": parent_id,
            "trainingType": training_type,
            "customPdfParse": custom_pdf_parse,
            "chunkSettingMode": chunk_setting_mode,
            "metadata": metadata or {}
        }
        
        # 仅在 custom 模式下添加分块参数
        if chunk_setting_mode == "custom":
            payload["chunkSplitMode"] = chunk_split_mode
            payload["chunkSize"] = chunk_size
            payload["indexSize"] = index_size
            if chunk_splitter:
                payload["chunkSplitter"] = chunk_splitter
        
        if training_type == "qa" and qa_prompt:
            payload["qaPrompt"] = qa_prompt
        if index_prefix_title is not None:
            payload["indexPrefixTitle"] = index_prefix_title
        if auto_indexes is not None:
            payload["autoIndexes"] = auto_indexes
        if image_index is not None:
            payload["imageIndex"] = image_index
        if tags:
            payload["tags"] = tags
        if create_time:
            payload["createTime"] = create_time
            
        res = self._post("/core/dataset/collection/create/text", json_data=payload)
        # API 返回的 data 直接是 collectionId 字符串
        return res['data']

    def create_link_collection(
        self,
        dataset_id: str,
        link: str,
        training_type: str = "chunk",  # chunk 或 qa
        parent_id: Optional[str] = None,
        index_prefix_title: Optional[bool] = True,
        custom_pdf_parse: bool = True,
        auto_indexes: Optional[bool] = True,
        image_index: Optional[bool] = True,
        chunk_setting_mode: str = "auto",  # auto 或 custom
        chunk_split_mode: str = "size",  # size 或 char
        chunk_size: int = 1500,
        index_size: int = 512,
        chunk_splitter: str = "",
        qa_prompt: str = "",
        selector: str = ".body",
        tags: Optional[List[str]] = None,
        create_time: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        导入链接数据 (URL)
        :return: 集合 ID (collectionId)
        """
        payload = {
            "datasetId": dataset_id,
            "link": link,
            "parentId": parent_id,
            "trainingType": training_type,
            "customPdfParse": custom_pdf_parse,
            "chunkSettingMode": chunk_setting_mode,
            "metadata": metadata or {"webPageSelector": selector}
        }
        
        # 仅在 custom 模式下添加分块参数
        if chunk_setting_mode == "custom":
            payload["chunkSplitMode"] = chunk_split_mode
            payload["chunkSize"] = chunk_size
            payload["indexSize"] = index_size
            if chunk_splitter:
                payload["chunkSplitter"] = chunk_splitter
        
        if training_type == "qa" and qa_prompt:
            payload["qaPrompt"] = qa_prompt
        if index_prefix_title is not None:
            payload["indexPrefixTitle"] = index_prefix_title
        if auto_indexes is not None:
            payload["autoIndexes"] = auto_indexes
        if image_index is not None:
            payload["imageIndex"] = image_index
        if tags:
            payload["tags"] = tags
        if create_time:
            payload["createTime"] = create_time
            
        res = self._post("/core/dataset/collection/create/link", json_data=payload)
        # API 返回的 data 直接是 collectionId 字符串
        return res['data']

    
    def create_file_collection(
        self,
        dataset_id: str,
        file_path: str,  # 支持本地文件路径或URL链接
        training_type: str = "chunk",  # chunk 或 qa
        name: Optional[str] = None,
        parent_id: Optional[str] = None,
        index_prefix_title: Optional[bool] = True,
        custom_pdf_parse: bool = True,
        auto_indexes: Optional[bool] = None,
        image_index: Optional[bool] = True,
        chunk_setting_mode: str = "auto",  # auto 或 custom
        chunk_split_mode: str = "size",  # size 或 char
        chunk_size: int = 1500,
        index_size: int = 512,
        chunk_splitter: str = "",
        qa_prompt: str = "",
        tags: Optional[List[str]] = None,
        create_time: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        导入文件 (PDF, Word, MD等)
        支持本地文件路径或URL链接，仅允许 pdf、xlsx、pptx、docx、md、txt 格式
        注意：使用 multipart/form-data，中文文件名需要 encode 处理
        :param file_path: 本地文件路径或URL链接
        :return: 集合 ID (collectionId)
        """
        # 允许的文件扩展名
        ALLOWED_EXTENSIONS = {'pdf', 'xlsx', 'pptx', 'docx', 'md', 'txt'}
        
        # 判断是否为URL
        is_url = file_path.startswith('http://') or file_path.startswith('https://')
        temp_file_path = None
        
        try:
            if is_url:
                # 从URL获取文件名
                parsed_url = urlparse(file_path)
                url_filename = os.path.basename(unquote(parsed_url.path))
                
                # 验证文件扩展名
                file_ext = url_filename.split('.')[-1].lower() if '.' in url_filename else ''
                if file_ext not in ALLOWED_EXTENSIONS:
                    raise ValueError(
                        f"不支持的文件格式: {file_ext}. "
                        f"仅允许以下格式: {', '.join(ALLOWED_EXTENSIONS)}"
                    )
                
                # 下载文件到临时目录
                response = requests.get(file_path, timeout=30)
                response.raise_for_status()
                
                # 创建临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
                    tmp_file.write(response.content)
                    temp_file_path = tmp_file.name
                
                actual_file_path = temp_file_path
                filename = url_filename
            else:
                # 本地文件路径
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"文件不存在: {file_path}")
                
                filename = os.path.basename(file_path)
                
                # 验证文件扩展名
                file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
                if file_ext not in ALLOWED_EXTENSIONS:
                    raise ValueError(
                        f"不支持的文件格式: {file_ext}. "
                        f"仅允许以下格式: {', '.join(ALLOWED_EXTENSIONS)}"
                    )
                
                actual_file_path = file_path
            
            # 对中文文件名进行 URL 编码
            encoded_filename = quote(name if name else filename, safe='')
            
            # 构造元数据
            form_metadata = {
                "datasetId": dataset_id,
                "parentId": parent_id,
                "trainingType": training_type,
                "customPdfParse": custom_pdf_parse,
                "chunkSettingMode": chunk_setting_mode,
                "metadata": metadata or {}
            }
            
            # 仅在 custom 模式下添加分块参数
            if chunk_setting_mode == "custom":
                form_metadata["chunkSplitMode"] = chunk_split_mode
                form_metadata["chunkSize"] = chunk_size
                form_metadata["indexSize"] = index_size
                if chunk_splitter:
                    form_metadata["chunkSplitter"] = chunk_splitter
            
            if training_type == "qa" and qa_prompt:
                form_metadata["qaPrompt"] = qa_prompt
            if index_prefix_title is not None:
                form_metadata["indexPrefixTitle"] = index_prefix_title
            if auto_indexes is not None:
                form_metadata["autoIndexes"] = auto_indexes
            if image_index is not None:
                form_metadata["imageIndex"] = image_index
            if tags:
                form_metadata["tags"] = tags
            if create_time:
                form_metadata["createTime"] = create_time
            
            # 上传文件
            url = f"{self.base_url}/core/dataset/collection/create/localFile"
            with open(actual_file_path, 'rb') as f:
                # 使用编码后的文件名
                files = {'file': (encoded_filename, f)}
                data = {'data': json.dumps(form_metadata)}
                
                # 特殊处理：Header 中去掉 Content-Type，让 requests 自动生成 boundary
                headers_copy = self.headers.copy()
                headers_copy.pop("Content-Type", None)
                
                response = requests.post(url, headers=headers_copy, files=files, data=data)
                response.raise_for_status()
                res_json = response.json()
                try:
                    return res_json['data']['collectionId']
                except:
                    return res_json['data']
        except Exception as e:
            raise Exception(f"File Upload Failed: {str(e)}")
        finally:
            # 清理临时文件
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

    # ---------------------- 数据追加 ----------------------

    def push_data(
        self,
        collection_id: str,
        data: List[Dict[str, str]], # [{"q": "...", "a": "..."}]
        training_type: str = "chunk"
    ) -> PushResults:
        """
        手动追加数据到集合
        """
        payload = {
            "collectionId": collection_id,
            "trainingType": training_type,
            "data": data
        }
        res = self._post("/core/dataset/data/pushData", json_data=payload)
        return PushResults(**res['data'])

    # ---------------------- 搜索测试 ----------------------

    def search_test(
        self,
        dataset_id: str,
        text: str,
        limit: int = 10,
        similarity: float = 0,
        using_re_rank: bool = False
    ) -> List[SearchResultItem]:
        """
        搜索测试
        :return: 搜索结果列表
        """
        payload = {
            "datasetId": dataset_id,
            "text": text,
            "limit": limit,
            "similarity": similarity,
            "usingReRank": using_re_rank,
            "searchMode": "embedding"
        }
        res = self._post("/core/dataset/searchTest", json_data=payload)
        # 结果是列表
        return SearchTestResult.from_list(res['data']['list']).list