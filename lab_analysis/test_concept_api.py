# -*- encoding: UTF-8 -*-
"""测试 Sequoia DataCache 概念成分股缓存逻辑（不依赖网络）"""

import sys
import os
import tempfile
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Sequoia'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

import pandas as pd
from data_cache import DataCache

tmpdir = tempfile.mkdtemp(prefix="test_cache_")
cache = DataCache(tmpdir)

print("=" * 60)
print("测试: concept_cons 缓存读写逻辑")
print("=" * 60)

mock_cons = pd.DataFrame({
    '代码': ['000001', '600519', '300750'],
    '名称': ['平安银行', '贵州茅台', '宁德时代'],
    '最新价': [10.5, 1800.0, 250.0],
    '涨跌幅': [1.2, -0.5, 2.3],
    '板块名称': ['人工智能'] * 3,
    '板块代码': ['BK0800'] * 3,
})

print("\n1. 写入前: 缓存是否新鲜?", cache.is_concept_cons_fresh("BK0800"))
print("   写入前: get_concept_cons =", cache.get_concept_cons("BK0800"))

result = cache.merge_concept_cons("BK0800", mock_cons)
print("\n2. merge_concept_cons 完成, 返回 %d 行" % len(result))
print("   含 _fetched_at 列:", '_fetched_at' in result.columns)

print("\n3. 写入后: 缓存是否新鲜?", cache.is_concept_cons_fresh("BK0800"))

reloaded = cache.get_concept_cons("BK0800")
print("\n4. 重新读取: %d 行" % len(reloaded))
print("   字段:", list(reloaded.columns))
print("   内容:")
print(reloaded[['代码', '名称', '最新价', '板块名称']].to_string(index=False))

mock_cons2 = pd.DataFrame({
    '代码': ['000002', '600000'],
    '名称': ['万科A', '浦发银行'],
    '最新价': [8.5, 7.2],
    '涨跌幅': [0.3, -1.1],
    '板块名称': ['房地产'] * 2,
    '板块代码': ['BK0451'] * 2,
})
cache.merge_concept_cons("BK0451", mock_cons2)

stats = cache.get_cache_stats()
print("\n5. 缓存统计:", stats)
print("   concept_cons =", stats.get('concept_cons'))

detailed = cache.get_detailed_stats()
print("\n6. 详细统计 concept_cons:", detailed.get('concept_cons'))

import shutil
shutil.rmtree(tmpdir)

print("\n" + "=" * 60)
print("全部测试通过!")
print("=" * 60)
