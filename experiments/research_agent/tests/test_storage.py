
# tests/test_storage.py
import sys
sys.path.append('..')

from storage.chroma_manager import ChromaManager

def test_chroma_init():
    """测试ChromaDB初始化"""
    print("=" * 60)
    print("测试ChromaDB初始化")
    print("=" * 60)
    
    try:
        # 初始化
        manager = ChromaManager(persist_directory="./data/chroma_db_test")
        
        # 获取统计信息
        stats = manager.get_statistics()
        print(f"\n✅ ChromaDB初始化成功！")
        print(f"论文数: {stats['total_papers']}")
        print(f"领域数: {stats['total_domains']}")
        print(f"用户配置: {'是' if stats['user_configured'] else '否'}")
        
        # 测试添加用户背景
        print("\n" + "-" * 60)
        print("测试设置用户背景")
        manager.set_user_background("我是一名软件工程师，熟悉Python和机器学习")
        
        bg = manager.get_user_background()
        print(f"✅ 用户背景设置成功: {bg}")
        
        # 测试添加论文
        print("\n" + "-" * 60)
        print("测试添加论文")
        manager.add_paper(
            paper_id="test_001",
            title="Test Paper on Machine Learning",
            abstract="This is a test paper about machine learning algorithms.",
            metadata={
                "authors": "Zhang San, Li Si",
                "year": 2024,
                "citation_count": 100,
                "paper_type": "classic",
                "domain": "machine_learning",
                "read_status": "unread"
            }
        )
        print("✅ 论文添加成功")
        
        # 测试语义搜索
        print("\n" + "-" * 60)
        print("测试语义搜索")
        results = manager.query_similar_papers(
            query_text="machine learning",
            n_results=1
        )
        print(f"✅ 找到 {len(results['ids'][0])} 篇相似论文")
        if results['ids'][0]:
            print(f"论文ID: {results['ids'][0][0]}")
            print(f"相似度距离: {results['distances'][0][0]}")
        
        # 测试添加领域
        print("\n" + "-" * 60)
        print("测试添加领域")
        manager.add_domain(
            domain_name="machine_learning",
            overview="机器学习是人工智能的一个分支...",
            metadata={
                "sub_fields": ["深度学习", "强化学习", "监督学习"],
                "key_concepts": ["神经网络", "梯度下降", "损失函数"],
                "current_stage": 1,
                "mode": "A"
            }
        )
        print("✅ 领域添加成功")
        
        # 获取领域信息
        domain_info = manager.get_domain("machine_learning")
        print(f"领域概述: {domain_info['overview'][:50]}...")
        print(f"当前阶段: {domain_info['metadata']['current_stage']}")
        
        print("\n" + "=" * 60)
        print("✅ 所有ChromaDB测试通过！")
        print("=" * 60)
        
        # 清理测试数据
        manager.reset()
        print("\n⚠️ 测试数据已清理")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chroma_init()
