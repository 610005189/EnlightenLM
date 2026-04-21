import os
import json
import time
import hashlib
import logging
import zlib
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import threading
import queue

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    审计日志系统，负责记录、压缩和验证日志
    """
    
    def __init__(self, config: Dict):
        """
        初始化审计日志系统
        
        Args:
            config: 日志配置
        """
        self.config = config
        self.storage_path = config.get("storage_path", "logs/audit")
        self.compression_config = config.get("compression", {})
        self.hash_chain_config = config.get("hash_chain", {})
        self.security_config = config.get("security", {})
        
        # 创建存储目录
        os.makedirs(self.storage_path, exist_ok=True)
        
        # 哈希算法
        self.hash_algorithm = self.hash_chain_config.get("algorithm", "sha256")
        self.initial_hash = self.hash_chain_config.get("initial_hash", "0" * 64)
        
        # 安全隔离配置
        self.security_enabled = self.security_config.get("enabled", False)
        self.security_type = self.security_config.get("type", "process")
        
        # 压缩配置
        self.compression_enabled = self.compression_config.get("enabled", True)
        self.compression_level = self.compression_config.get("level", 6)
        
        # 日志队列
        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # 启动日志处理线程
        self.log_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.log_thread.start()
        
        # 加载上一次的哈希
        self.last_log_hash = self._load_last_hash()
    
    def _load_last_hash(self) -> str:
        """
        加载上一次的日志哈希
        
        Returns:
            上一次的日志哈希
        """
        hash_file = os.path.join(self.storage_path, "last_hash.txt")
        if os.path.exists(hash_file):
            try:
                with open(hash_file, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                logger.error(f"Failed to load last hash: {e}")
        return self.initial_hash
    
    def _save_last_hash(self, hash_value: str):
        """
        保存上一次的日志哈希
        
        Args:
            hash_value: 哈希值
        """
        hash_file = os.path.join(self.storage_path, "last_hash.txt")
        try:
            with open(hash_file, 'w') as f:
                f.write(hash_value)
        except Exception as e:
            logger.error(f"Failed to save last hash: {e}")
    
    def _compress_data(self, data: bytes) -> bytes:
        """
        压缩数据
        
        Args:
            data: 原始数据
            
        Returns:
            压缩后的数据
        """
        if self.compression_enabled:
            return zlib.compress(data, level=self.compression_level)
        return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """
        解压数据
        
        Args:
            data: 压缩数据
            
        Returns:
            解压后的数据
        """
        if self.compression_enabled:
            return zlib.decompress(data)
        return data
    
    def _process_logs(self):
        """
        处理日志队列
        """
        while not self.stop_event.is_set():
            try:
                log_entry = self.log_queue.get(timeout=1)
                self._write_log(log_entry)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing log: {e}")
    
    def _write_log(self, log_entry: Dict):
        """
        写入日志
        
        Args:
            log_entry: 日志条目
        """
        # 计算日志哈希
        log_content = json.dumps(log_entry, sort_keys=True)
        log_hash = hashlib.new(self.hash_algorithm, log_content.encode()).hexdigest()
        
        # 添加哈希链信息
        log_entry["prev_log_hash"] = self.last_log_hash
        log_entry["log_hash"] = log_hash
        
        # 更新最后哈希
        self.last_log_hash = log_hash
        self._save_last_hash(log_hash)
        
        # 确定日志文件路径
        timestamp = log_entry.get("timestamp_us", int(time.time() * 1_000_000))
        dt = datetime.fromtimestamp(timestamp / 1_000_000)
        file_path = os.path.join(
            self.storage_path,
            f"{dt.year}",
            f"{dt.month:02d}",
            f"{dt.day:02d}"
        )
        os.makedirs(file_path, exist_ok=True)
        
        # 写入日志文件
        file_name = f"{dt.hour:02d}.jsonl"
        full_path = os.path.join(file_path, file_name)
        
        try:
            with open(full_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write log: {e}")
        
        # 每小时生成Merkle树
        if dt.minute == 0 and dt.second == 0:
            self._generate_merkle_tree(dt.year, dt.month, dt.day, dt.hour)
    
    def _generate_merkle_tree(self, year: int, month: int, day: int, hour: int):
        """
        生成Merkle树
        
        Args:
            year: 年份
            month: 月份
            day: 日期
            hour: 小时
        """
        file_path = os.path.join(
            self.storage_path,
            f"{year}",
            f"{month:02d}",
            f"{day:02d}"
        )
        file_name = f"{hour:02d}.jsonl"
        full_path = os.path.join(file_path, file_name)
        
        if not os.path.exists(full_path):
            return
        
        # 读取日志
        logs = []
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error reading log file {full_path}: {e}")
            return
        
        if not logs:
            return
        
        # 生成Merkle树
        merkle_root = self._build_merkle_tree([log.get("log_hash") for log in logs])
        
        # 保存Merkle树信息
        merkle_info = {
            "timestamp": int(time.time() * 1_000_000),
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "merkle_root": merkle_root,
            "log_count": len(logs)
        }
        
        merkle_file = os.path.join(file_path, f"{hour:02d}_merkle.json")
        try:
            with open(merkle_file, 'w', encoding='utf-8') as f:
                json.dump(merkle_info, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save merkle tree: {e}")
    
    def _build_merkle_tree(self, hashes: List[str]) -> str:
        """
        构建Merkle树
        
        Args:
            hashes: 哈希列表
            
        Returns:
            Merkle根哈希
        """
        if not hashes:
            return self.initial_hash
        
        if len(hashes) == 1:
            return hashes[0]
        
        # 确保哈希数量为偶数
        if len(hashes) % 2 != 0:
            hashes.append(hashes[-1])
        
        new_hashes = []
        for i in range(0, len(hashes), 2):
            combined = hashes[i] + hashes[i+1]
            new_hash = hashlib.new(self.hash_algorithm, combined.encode()).hexdigest()
            new_hashes.append(new_hash)
        
        return self._build_merkle_tree(new_hashes)
    
    def log_session(self, session_id: str, input_text: str, output_text: str, 
                    meta_description: str, core_rules: Dict, user_params: Dict, 
                    attention_stats: Dict, cutoff_event: Dict):
        """
        记录会话日志
        
        Args:
            session_id: 会话ID
            input_text: 输入文本
            output_text: 输出文本
            meta_description: 元描述
            core_rules: 核心规则
            user_params: 用户参数
            attention_stats: 注意力统计
            cutoff_event: 截断事件
        """
        # 计算输入和输出的哈希
        input_hash = hashlib.new(self.hash_algorithm, input_text.encode()).hexdigest()
        output_hash = hashlib.new(self.hash_algorithm, output_text.encode()).hexdigest()
        meta_description_hash = hashlib.new(self.hash_algorithm, meta_description.encode()).hexdigest()
        
        # 构建日志条目
        log_entry = {
            "session_id": session_id,
            "timestamp_us": int(time.time() * 1_000_000),
            "input_hash": input_hash,
            "output_hash": output_hash,
            "core_rules_bitmap": core_rules.get("bitmap", 0),
            "vocab_version_hash": core_rules.get("vocab_version_hash", ""),
            "user_params_safe": list(user_params.values()),
            "attn_stats": {
                "bias_norm_quant": attention_stats.get("bias_norm_quant", 0),
                "core_overlap_quant": attention_stats.get("core_overlap_quant", 0),
                "entropy_quant": attention_stats.get("entropy_quant", 0)
            },
            "cutoff_depth": cutoff_event.get("depth", 0),
            "cutoff_reason": cutoff_event.get("reason", ""),
            "snapshot_hash": cutoff_event.get("snapshot_hash", ""),
            "meta_description_hash": meta_description_hash
        }
        
        # 加入队列
        self.log_queue.put(log_entry)
    
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """
        获取会话摘要
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话摘要
        """
        # 搜索会话日志
        for root, dirs, files in os.walk(self.storage_path):
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    log_entry = json.loads(line)
                                    if log_entry.get("session_id") == session_id:
                                        return {
                                            "session_id": log_entry.get("session_id"),
                                            "timestamp": log_entry.get("timestamp_us"),
                                            "input_hash": log_entry.get("input_hash"),
                                            "output_hash": log_entry.get("output_hash"),
                                            "cutoff": log_entry.get("cutoff_reason", "") != "",
                                            "cutoff_reason": log_entry.get("cutoff_reason", "")
                                        }
                                except json.JSONDecodeError:
                                    continue
                    except Exception as e:
                        logger.error(f"Error reading log file {file_path}: {e}")
        
        return None
    
    def verify_hash_chain(self) -> bool:
        """
        验证哈希链的完整性
        
        Returns:
            哈希链是否完整
        """
        # 遍历所有日志文件
        logs = []
        for root, dirs, files in os.walk(self.storage_path):
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    log_entry = json.loads(line)
                                    logs.append(log_entry)
                                except json.JSONDecodeError:
                                    continue
                    except Exception as e:
                        logger.error(f"Error reading log file {file_path}: {e}")
        
        # 按时间戳排序
        logs.sort(key=lambda x: x.get("timestamp_us", 0))
        
        # 验证哈希链
        previous_hash = self.initial_hash
        for log in logs:
            # 移除哈希字段以计算原始哈希
            log_copy = log.copy()
            current_hash = log_copy.pop("log_hash")
            prev_hash = log_copy.pop("prev_log_hash")
            
            # 验证前哈希
            if prev_hash != previous_hash:
                logger.error(f"Hash chain broken at log {log.get('session_id')}")
                return False
            
            # 计算当前哈希并验证
            log_content = json.dumps(log_copy, sort_keys=True)
            computed_hash = hashlib.new(self.hash_algorithm, log_content.encode()).hexdigest()
            if computed_hash != current_hash:
                logger.error(f"Hash mismatch at log {log.get('session_id')}")
                return False
            
            previous_hash = current_hash
        
        # 验证最后哈希
        if previous_hash != self.last_log_hash:
            logger.error("Last hash mismatch")
            return False
        
        # 验证Merkle树
        self._verify_merkle_trees()
        
        return True
    
    def _verify_merkle_trees(self):
        """
        验证Merkle树
        """
        for root, dirs, files in os.walk(self.storage_path):
            for file in files:
                if file.endswith('_merkle.json'):
                    merkle_file = os.path.join(root, file)
                    try:
                        with open(merkle_file, 'r', encoding='utf-8') as f:
                            merkle_info = json.load(f)
                        
                        # 读取对应小时的日志
                        hour_file = os.path.join(root, file.replace('_merkle.json', '.jsonl'))
                        if not os.path.exists(hour_file):
                            logger.error(f"Hourly log file missing for merkle tree: {hour_file}")
                            continue
                        
                        logs = []
                        with open(hour_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    log_entry = json.loads(line)
                                    logs.append(log_entry)
                                except json.JSONDecodeError:
                                    continue
                        
                        # 验证Merkle根
                        expected_root = merkle_info.get("merkle_root")
                        actual_root = self._build_merkle_tree([log.get("log_hash") for log in logs])
                        if expected_root != actual_root:
                            logger.error(f"Merkle tree mismatch for {merkle_file}")
                    except Exception as e:
                        logger.error(f"Error verifying merkle tree {merkle_file}: {e}")
    
    def export_logs(self, start_time: int, end_time: int) -> Optional[str]:
        """
        导出日志
        
        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳
            
        Returns:
            导出文件路径
        """
        # 收集日志
        logs = []
        for root, dirs, files in os.walk(self.storage_path):
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    log_entry = json.loads(line)
                                    timestamp = log_entry.get("timestamp_us", 0)
                                    if start_time <= timestamp <= end_time:
                                        logs.append(log_entry)
                                except json.JSONDecodeError:
                                    continue
                    except Exception as e:
                        logger.error(f"Error reading log file {file_path}: {e}")
        
        if not logs:
            return None
        
        # 排序日志
        logs.sort(key=lambda x: x.get("timestamp_us", 0))
        
        # 生成导出文件
        export_file = os.path.join(self.storage_path, f"export_{start_time}_{end_time}.jsonl")
        try:
            with open(export_file, 'w', encoding='utf-8') as f:
                for log in logs:
                    f.write(json.dumps(log, ensure_ascii=False) + '\n')
            
            # 生成导出文件的哈希
            export_hash = hashlib.new(self.hash_algorithm, open(export_file, 'rb').read()).hexdigest()
            
            # 保存导出信息
            export_info = {
                "export_time": int(time.time() * 1_000_000),
                "start_time": start_time,
                "end_time": end_time,
                "log_count": len(logs),
                "export_hash": export_hash
            }
            
            info_file = os.path.join(self.storage_path, f"export_{start_time}_{end_time}_info.json")
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(export_info, f, ensure_ascii=False)
            
            return export_file
        except Exception as e:
            logger.error(f"Failed to export logs: {e}")
            return None
    
    def shutdown(self):
        """
        关闭审计日志系统
        """
        self.stop_event.set()
        self.log_thread.join(timeout=5)
