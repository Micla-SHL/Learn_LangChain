
# storage/chroma_manager.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from datetime import datetime
import json

class ChromaManager:
    """ChromaDB存储管理器"""
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        初始化ChromaDB
        
        Args:
            persist_directory: 数据持久化目录
        """
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # 初始化collections
        self._init_collections()
    
    def _init_collections(self):
        """初始化所有collections"""
        
        # Collection 1: papers - 存储论文信息
        self.papers_collection = self.client.get_or_create_collection(
            name="papers",
            metadata={"description": "存储论文信息和摘要"}
        )
        
        # Collection 2: domains - 存储领域信息
        self.domains_collection = self.client.get_or_create_collection(
            name="domains",
            metadata={"description": "存储领域知识框架"}
        )
        
        # Collection 3: user_profile - 存储用户配置
        self.user_profile_collection = self.client.get_or_create_collection(
            name="user_profile",
            metadata={"description": "存储用户背景信息"}
        )
        
        print("✅ ChromaDB collections 初始化成功")
    
    # ==================== 论文相关操作 ====================
    
    def add_paper(
        self,
        paper_id: str,
        title: str,
        abstract: str,  # 用于embedding
        metadata: Dict
    ):
        """
        添加论文到数据库
        
        Args:
            paper_id: 论文唯一ID
            title: 标题
            abstract: 摘要（用于向量化）
            metadata: 其他元数据（作者、年份、引用数等）
        """
        self.papers_collection.add(
            ids=[paper_id],
            documents=[abstract],  # 摘要用于生成向量
            metadatas=[{
                "title": title,
                "added_date": datetime.now().isoformat(),
                **metadata
            }]
        )
    
    def query_similar_papers(
        self,
        query_text: str,
        domain: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict]:
        """
        语义搜索相似论文（避免重复）
        
        Args:
            query_text: 查询文本（论文标题或摘要）
            domain: 限定领域
            n_results: 返回数量
            
        Returns:
            相似论文列表
        """
        where = {"domain": domain} if domain else None
        
        results = self.papers_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def get_papers_by_domain(
        self,
        domain: str,
        paper_type: Optional[str] = None
    ) -> List[Dict]:
        """
        获取某领域的所有论文
        
        Args:
            domain: 领域名
            paper_type: 论文类型筛选（review/classic/frontier）
        """
        where = {"domain": domain}
        if paper_type:
            where["paper_type"] = paper_type
        
        results = self.papers_collection.get(where=where)
        return results
    
    def update_paper_status(self, paper_id: str, status: str):
        """
        更新论文阅读状态
        
        Args:
            paper_id: 论文ID
            status: 状态（unread/reading/completed）
        """
        self.papers_collection.update(
            ids=[paper_id],
            metadatas=[{"read_status": status}]
        )
    
    # ==================== 领域相关操作 ====================
    
    def add_domain(
        self,
        domain_name: str,
        overview: str,
        metadata: Dict
    ):
        """
        添加领域信息
        
        Args:
            domain_name: 领域名称
            overview: 领域概述
            metadata: 其他元数据（子领域、核心概念等）
        """
        self.domains_collection.add(
            ids=[domain_name],
            documents=[overview],
            metadatas=[{
                "created_date": datetime.now().isoformat(),
                **metadata
            }]
        )
    
    def get_domain(self, domain_name: str) -> Optional[Dict]:
        """获取领域信息"""
        results = self.domains_collection.get(ids=[domain_name])
        
        if results["ids"]:
            return {
                "id": results["ids"][0],
                "overview": results["documents"][0],
                "metadata": results["metadatas"][0]
            }
        return None
    
    def update_domain_stage(self, domain_name: str, stage: int):
        """更新领域学习阶段"""
        self.domains_collection.update(
            ids=[domain_name],
            metadatas=[{"current_stage": stage}]
        )
    
    # ==================== 用户配置操作 ====================
    
    def set_user_background(self, background: str):
        """设置用户背景（全局）"""
        user_id = "global_user"  # 单用户系统
        
        # 检查是否已存在
        existing = self.user_profile_collection.get(ids=[user_id])
        
        if existing["ids"]:
            # 更新
            self.user_profile_collection.update(
                ids=[user_id],
                documents=[background],
                metadatas=[{"updated_date": datetime.now().isoformat()}]
            )
        else:
            # 创建
            self.user_profile_collection.add(
                ids=[user_id],
                documents=[background],
                metadatas=[{"created_date": datetime.now().isoformat()}]
            )
    
    def get_user_background(self) -> Optional[str]:
        """获取用户背景"""
        user_id = "global_user"
        results = self.user_profile_collection.get(ids=[user_id])
        
        if results["ids"]:
            return results["documents"][0]
        return None
    
    # ==================== 统计信息 ====================
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "total_papers": self.papers_collection.count(),
            "total_domains": self.domains_collection.count(),
            "user_configured": self.user_profile_collection.count() > 0
        }
    
    def reset(self):
        """重置所有数据（慎用）"""
        self.client.delete_collection("papers")
        self.client.delete_collection("domains")
        self.client.delete_collection("user_profile")
        self._init_collections()
        print("⚠️ 所有数据已重置")


# 创建全局实例
chroma_manager = ChromaManager()
