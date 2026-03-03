# -*- encoding: UTF-8 -*-

"""隧道代理池 — 透明 monkey-patch requests，让 akshare 自动走代理

原理：
    替换 requests.Session.request，仅对东财/新浪等目标域名注入代理。
    akshare 源码零修改，升级无忧。

使用：
    from proxy_pool import ProxyPool
    pool = ProxyPool(config['proxy'])
    pool.install()      # 安装 monkey-patch
    # ... 正常调用 akshare ...
    pool.uninstall()    # 卸载（可选）
"""

import logging
import re
from urllib.parse import urlparse

import requests

_TARGET_DOMAINS = re.compile(
    r'(eastmoney\.com|sina\.com\.cn|sina\.cn|sinaimg\.cn'
    r'|mairui\.club|10jqka\.com\.cn'
    r'|push2\.eastmoney|push2ex\.eastmoney|push2his\.eastmoney'
    r'|emweb\.securities\.eastmoney)',
    re.IGNORECASE,
)


class ProxyPool:
    """基于快代理隧道的透明代理池。

    Parameters
    ----------
    proxy_cfg : dict
        来自 stock_config.yaml 的 proxy 段，包含：
        tunnel, tunnel_backup, username, password, timeout, max_retries
    """

    def __init__(self, proxy_cfg: dict):
        self._cfg = proxy_cfg
        self._timeout = proxy_cfg.get('timeout', 10)
        self._max_retries = proxy_cfg.get('max_retries', 3)

        tunnel = proxy_cfg['tunnel']
        tunnel_backup = proxy_cfg.get('tunnel_backup', '')
        user = proxy_cfg['username']
        pwd = proxy_cfg['password']

        self._proxy_main = {
            "http": f"http://{user}:{pwd}@{tunnel}/",
            "https": f"http://{user}:{pwd}@{tunnel}/",
        }
        self._proxy_backup = None
        if tunnel_backup:
            self._proxy_backup = {
                "http": f"http://{user}:{pwd}@{tunnel_backup}/",
                "https": f"http://{user}:{pwd}@{tunnel_backup}/",
            }

        self._installed = False
        self._original_request = None

    def _should_proxy(self, url: str) -> bool:
        try:
            host = urlparse(url).hostname or ''
        except Exception:
            return False
        return bool(_TARGET_DOMAINS.search(host))

    def install(self):
        """安装 monkey-patch，对目标域名自动注入代理。"""
        if self._installed:
            return
        self._original_request = requests.Session.request
        pool = self

        def _patched_request(session_self, method, url, **kwargs):
            if not pool._should_proxy(url):
                return pool._original_request(session_self, method, url, **kwargs)

            # 禁止 keep-alive 复用连接（隧道代理每次请求换 IP）
            kwargs.setdefault('headers', {})
            if isinstance(kwargs['headers'], dict):
                kwargs['headers']['Connection'] = 'close'

            proxies_chain = [pool._proxy_main]
            if pool._proxy_backup:
                proxies_chain.append(pool._proxy_backup)

            last_err = None
            for proxy_dict in proxies_chain:
                for attempt in range(pool._max_retries):
                    kw = dict(kwargs)
                    kw['proxies'] = proxy_dict
                    kw.setdefault('timeout', pool._timeout)
                    try:
                        resp = pool._original_request(
                            session_self, method, url, **kw)
                        if resp.status_code in (403, 407, 429, 456, 503):
                            logging.debug(
                                "ProxyPool: %s 返回 %d，重试 (%d/%d)",
                                url[:60], resp.status_code,
                                attempt + 1, pool._max_retries)
                            last_err = None
                            continue
                        return resp
                    except (requests.exceptions.ProxyError,
                            requests.exceptions.ConnectionError,
                            requests.exceptions.Timeout) as e:
                        last_err = e
                        logging.debug(
                            "ProxyPool: 代理失败 (%d/%d) %s: %s",
                            attempt + 1, pool._max_retries,
                            url[:60], type(e).__name__)
                        continue

            # 所有代理都失败，回退直连
            logging.warning(
                "ProxyPool: 代理均失败，回退直连 %s (最后错误: %s)",
                url[:60], last_err)
            kwargs.pop('proxies', None)
            return pool._original_request(session_self, method, url, **kwargs)

        requests.Session.request = _patched_request
        self._installed = True
        logging.info("ProxyPool: monkey-patch 已安装 (隧道: %s)",
                     self._cfg.get('tunnel', ''))

    def uninstall(self):
        """卸载 monkey-patch，恢复原始 requests。"""
        if self._installed and self._original_request:
            requests.Session.request = self._original_request
            self._installed = False
            logging.info("ProxyPool: monkey-patch 已卸载")
