import os
import logging
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, Optional
import threading
import queue
import tempfile
import stat

logger = logging.getLogger(__name__)


class SecurityIsolation:
    """
    安全隔离模块，用于实现日志的硬件级防篡改
    """
    
    def __init__(self, config: Dict):
        """
        初始化安全隔离模块
        
        Args:
            config: 安全配置
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        self.isolation_type = config.get("type", "process")
        self.storage_path = config.get("storage_path", "logs/secure")
        self.hash_algorithm = config.get("hash_algorithm", "sha256")
        
        # 创建存储目录
        os.makedirs(self.storage_path, exist_ok=True)
        
        # 设置目录权限为只读
        os.chmod(self.storage_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        
        # 日志队列
        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # 启动隔离线程
        if self.enabled:
            self.isolation_thread = threading.Thread(target=self._isolation_loop, daemon=True)
            self.isolation_thread.start()
            logger.info(f"Security isolation enabled with type: {self.isolation_type}")
        else:
            logger.info("Security isolation disabled")
    
    def _isolation_loop(self):
        """
        隔离循环
        """
        while not self.stop_event.is_set():
            try:
                log_entry = self.log_queue.get(timeout=1)
                self._process_log(log_entry)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in isolation loop: {e}")
    
    def _process_log(self, log_entry: Dict):
        """
        处理日志
        
        Args:
            log_entry: 日志条目
        """
        if self.isolation_type == "process":
            # 进程级隔离
            self._process_isolation(log_entry)
        elif self.isolation_type == "sgx":
            # SGX隔离
            self._sgx_isolation(log_entry)
        elif self.isolation_type == "sev":
            # SEV隔离
            self._sev_isolation(log_entry)
        else:
            logger.error(f"Unknown isolation type: {self.isolation_type}")
    
    def _process_isolation(self, log_entry: Dict):
        """
        进程级隔离
        
        Args:
            log_entry: 日志条目
        """
        try:
            # 计算日志哈希
            log_content = json.dumps(log_entry, sort_keys=True)
            log_hash = hashlib.new(self.hash_algorithm, log_content.encode()).hexdigest()
            
            # 确定日志文件路径
            timestamp = log_entry.get("timestamp_us", int(time.time() * 1_000_000))
            dt = datetime.fromtimestamp(timestamp / 1_000_000)
            file_path = os.path.join(
                self.storage_path,
                f"{dt.year}",
                f"{dt.month:02d}",
                f"{dt.day:02d}"
            )
            
            # 创建目录（如果不存在）
            os.makedirs(file_path, exist_ok=True)
            os.chmod(file_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            
            # 写入临时文件
            temp_file = tempfile.NamedTemporaryFile(dir=file_path, delete=False)
            with temp_file:
                temp_file.write(json.dumps(log_entry, ensure_ascii=False).encode())
            
            # 重命名为最终文件
            file_name = f"{dt.hour:02d}_{log_hash[:16]}.json"
            full_path = os.path.join(file_path, file_name)
            os.rename(temp_file.name, full_path)
            
            # 设置文件权限为只读
            os.chmod(full_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            
            logger.debug(f"Process isolation: Log written to {full_path}")
        except Exception as e:
            logger.error(f"Error in process isolation: {e}")
    
    def _sgx_isolation(self, log_entry: Dict):
        """
        SGX隔离
        
        Args:
            log_entry: 日志条目
        """
        try:
            # 这里实现SGX隔离逻辑
            # 由于SGX需要特殊的硬件和SDK，这里提供一个模拟实现
            logger.debug(f"SGX isolation: Processing log {log_entry.get('session_id')}")
            
            # 模拟SGX enclave操作
            # 实际实现需要使用Intel SGX SDK
            self._simulate_sgx_operation(log_entry)
        except Exception as e:
            logger.error(f"Error in SGX isolation: {e}")
    
    def _sev_isolation(self, log_entry: Dict):
        """
        SEV隔离
        
        Args:
            log_entry: 日志条目
        """
        try:
            # 这里实现SEV隔离逻辑
            # 由于SEV需要特殊的硬件和SDK，这里提供一个模拟实现
            logger.debug(f"SEV isolation: Processing log {log_entry.get('session_id')}")
            
            # 模拟SEV虚拟机操作
            # 实际实现需要使用AMD SEV SDK
            self._simulate_sev_operation(log_entry)
        except Exception as e:
            logger.error(f"Error in SEV isolation: {e}")
    
    def _simulate_sgx_operation(self, log_entry: Dict):
        """
        模拟SGX操作
        
        Args:
            log_entry: 日志条目
        """
        # 模拟SGX enclave中的操作
        # 实际实现需要使用Intel SGX SDK
        logger.debug("Simulating SGX operation")
        
        # 计算日志哈希
        log_content = json.dumps(log_entry, sort_keys=True)
        log_hash = hashlib.new(self.hash_algorithm, log_content.encode()).hexdigest()
        
        # 写入模拟的SGX日志
        sgx_path = os.path.join(self.storage_path, "sgx")
        os.makedirs(sgx_path, exist_ok=True)
        os.chmod(sgx_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        
        file_name = f"sgx_{log_hash[:16]}.json"
        full_path = os.path.join(sgx_path, file_name)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False)
        
        os.chmod(full_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    
    def _simulate_sev_operation(self, log_entry: Dict):
        """
        模拟SEV操作
        
        Args:
            log_entry: 日志条目
        """
        # 模拟SEV虚拟机中的操作
        # 实际实现需要使用AMD SEV SDK
        logger.debug("Simulating SEV operation")
        
        # 计算日志哈希
        log_content = json.dumps(log_entry, sort_keys=True)
        log_hash = hashlib.new(self.hash_algorithm, log_content.encode()).hexdigest()
        
        # 写入模拟的SEV日志
        sev_path = os.path.join(self.storage_path, "sev")
        os.makedirs(sev_path, exist_ok=True)
        os.chmod(sev_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        
        file_name = f"sev_{log_hash[:16]}.json"
        full_path = os.path.join(sev_path, file_name)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False)
        
        os.chmod(full_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    
    def add_log(self, log_entry: Dict):
        """
        添加日志到隔离区
        
        Args:
            log_entry: 日志条目
        """
        if self.enabled:
            self.log_queue.put(log_entry)
        else:
            # 如果未启用隔离，直接返回
            pass
    
    def verify_secure_logs(self) -> bool:
        """
        验证安全日志的完整性
        
        Returns:
            日志是否完整
        """
        if not self.enabled:
            return False
        
        try:
            # 遍历所有安全日志文件
            for root, dirs, files in os.walk(self.storage_path):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        
                        # 检查文件权限
                        if not os.access(file_path, os.R_OK):
                            logger.error(f"Cannot read secure log file: {file_path}")
                            return False
                        
                        # 验证文件内容
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                log_entry = json.load(f)
                            
                            # 验证日志哈希
                            log_copy = log_entry.copy()
                            if 'log_hash' in log_copy:
                                log_copy.pop('log_hash')
                            if 'prev_log_hash' in log_copy:
                                log_copy.pop('prev_log_hash')
                            
                            log_content = json.dumps(log_copy, sort_keys=True)
                            computed_hash = hashlib.new(self.hash_algorithm, log_content.encode()).hexdigest()
                            
                            if 'log_hash' in log_entry and log_entry['log_hash'] != computed_hash:
                                logger.error(f"Log hash mismatch: {file_path}")
                                return False
                        except Exception as e:
                            logger.error(f"Error verifying secure log {file_path}: {e}")
                            return False
            
            return True
        except Exception as e:
            logger.error(f"Error verifying secure logs: {e}")
            return False
    
    def shutdown(self):
        """
        关闭安全隔离模块
        """
        if self.enabled:
            self.stop_event.set()
            self.isolation_thread.join(timeout=5)
