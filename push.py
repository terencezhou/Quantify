# -*- encoding: UTF-8 -*-

import logging

_push_config = None


def init(config: dict):
    """用外部传入的配置初始化推送模块（替代依赖 settings 全局变量）。"""
    global _push_config
    _push_config = config.get('push', {})


def _get_config():
    global _push_config
    if _push_config is not None:
        return _push_config
    try:
        import settings
        cfg = settings.config if callable(settings.config) else settings.config
        if isinstance(cfg, dict):
            _push_config = cfg.get('push', {})
            return _push_config
    except Exception:
        pass
    _push_config = {}
    return _push_config


def _send(msg, content_type=1):
    cfg = _get_config()
    if cfg.get('enable', False):
        try:
            from wxpusher import WxPusher
            response = WxPusher.send_message(
                msg,
                uids=[cfg['wxpusher_uid']],
                token=cfg['wxpusher_token'],
                content_type=content_type,
            )
            logging.info("推送结果: %s", response)
        except Exception as e:
            logging.warning("推送失败: %s", e)
    logging.info(msg)


def markdown(msg):
    _send(msg, content_type=3)


def text(msg):
    _send(msg, content_type=1)
