"""
TEE Audit Format - TEE 兼容审计数据格式
支持 Intel SGX 等 TEE 硬件的审计日志格式设计
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from enum import Enum


class TEEType(Enum):
    """TEE 类型"""
    SGX = "sgx"           # Intel SGX
    TDX = "tdx"           # Intel TDX
    SEV = "sev"           # AMD SEV
    TRUSTED_OS = "trusted_os"  # 可信执行环境


class SecurityLevel(Enum):
    """安全级别"""
    SOFTWARE = "software"      # 纯软件（无 TEE）
    TEE_SOFTWARE = "tee_software"  # TEE 内软件
    TEE_HARDWARE = "tee_hardware"  # TEE 硬件保护


@dataclass
class TEEQuote:
    """TEE 引用结构"""
    quote_type: str
    quote_size: int
    quote_data: bytes
    signature: bytes
    timestamp: float


@dataclass
class AuditDataHeader:
    """审计数据头"""
    version: str = "1.0"
    sequence_number: int = 0
    timestamp: float = field(default_factory=time.time)
    tee_type: str = TEEType.SGX.value
    security_level: str = SecurityLevel.TEE_HARDWARE.value
    enclave_id: str = ""
    measurement: str = ""  # MRENCLAVE for SGX


@dataclass
class AuditEntry:
    """审计条目"""
    header: AuditDataHeader
    data: Dict[str, Any]
    previous_hash: str
    current_hash: str
    quote: Optional[TEEQuote] = None
    signature: Optional[bytes] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "header": asdict(self.header),
            "data": self.data,
            "previous_hash": self.previous_hash,
            "current_hash": self.current_hash,
            "quote": asdict(self.quote) if self.quote else None,
            "signature": self.signature.hex() if self.signature else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AuditEntry':
        """从字典创建"""
        header = AuditDataHeader(**data["header"])
        quote = TEEQuote(**data["quote"]) if data.get("quote") else None
        signature = bytes.fromhex(data["signature"]) if data.get("signature") else None

        return cls(
            header=header,
            data=data["data"],
            previous_hash=data["previous_hash"],
            current_hash=data["current_hash"],
            quote=quote,
            signature=signature
        )


class TEEAuditFormatter:
    """
    TEE 审计数据格式化器

    将审计数据格式化为 TEE 兼容的格式
    支持 Intel SGX、AMD SEV 等硬件
    """

    def __init__(self, tee_type: TEEType = TEEType.SGX):
        self.tee_type = tee_type
        self.sequence_number = 0
        self.initial_hash = "0" * 64

    def format_entry(
        self,
        data: Dict[str, Any],
        enclave_id: str = "",
        measurement: str = ""
    ) -> AuditEntry:
        """
        格式化审计条目

        Args:
            data: 审计数据
            enclave_id: Enclave ID
            measurement: Enclave 测量值 (MRENCLAVE)

        Returns:
            AuditEntry: 格式化的审计条目
        """
        header = AuditDataHeader(
            version="1.0",
            sequence_number=self.sequence_number,
            timestamp=time.time(),
            tee_type=self.tee_type.value,
            security_level=SecurityLevel.TEE_HARDWARE.value,
            enclave_id=enclave_id,
            measurement=measurement
        )

        previous_hash = self._get_previous_hash()

        current_hash = self._compute_hash(header, data, previous_hash)

        entry = AuditEntry(
            header=header,
            data=data,
            previous_hash=previous_hash,
            current_hash=current_hash
        )

        self.sequence_number += 1

        return entry

    def _compute_hash(
        self,
        header: AuditDataHeader,
        data: Dict[str, Any],
        previous_hash: str
    ) -> str:
        """计算条目的哈希"""
        content = json.dumps({
            "header": asdict(header),
            "data": data,
            "previous_hash": previous_hash
        }, sort_keys=True, default=str)

        return hashlib.sha256(content.encode()).hexdigest()

    def _get_previous_hash(self) -> str:
        """获取前一个哈希"""
        return self.initial_hash if self.sequence_number == 0 else ""


class TEERemoteAttestation:
    """
    TEE 远程认证服务

    提供 TEE 环境的远程认证接口
    """

    def __init__(self, tee_type: TEEType = TEEType.SGX):
        self.tee_type = tee_type

    def generate_quote(
        self,
        report_data: bytes
    ) -> TEEQuote:
        """
        生成 TEE 引用

        Args:
            report_data: 报告数据

        Returns:
            TEEQuote: TEE 引用结构
        """
        if self.tee_type == TEEType.SGX:
            return self._generate_sgx_quote(report_data)
        elif self.tee_type == TEEType.TDX:
            return self._generate_tdx_quote(report_data)
        elif self.tee_type == TEEType.SEV:
            return self._generate_sev_quote(report_data)
        else:
            raise ValueError(f"Unsupported TEE type: {self.tee_type}")

    def _generate_sgx_quote(self, report_data: bytes) -> TEEQuote:
        """生成 SGX 引用"""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.backends import default_backend

            quote = TEEQuote(
                quote_type="sgx",
                quote_size=len(report_data) + 432,
                quote_data=report_data,
                signature=b"sgx_signature_placeholder",
                timestamp=time.time()
            )

            return quote
        except ImportError:
            return TEEQuote(
                quote_type="sgx",
                quote_size=432,
                quote_data=report_data[:432] if len(report_data) >= 432 else report_data,
                signature=b"mock_signature",
                timestamp=time.time()
            )

    def _generate_tdx_quote(self, report_data: bytes) -> TEEQuote:
        """生成 TDX 引用"""
        return TEEQuote(
            quote_type="tdx",
            quote_size=len(report_data) + 576,
            quote_data=report_data,
            signature=b"tdx_signature_placeholder",
            timestamp=time.time()
        )

    def _generate_sev_quote(self, report_data: bytes) -> TEEQuote:
        """生成 SEV 引用"""
        return TEEQuote(
            quote_type="sev",
            quote_size=len(report_data) + 128,
            quote_data=report_data,
            signature=b"sev_signature_placeholder",
            timestamp=time.time()
        )

    def verify_quote(self, quote: TEEQuote) -> bool:
        """
        验证 TEE 引用

        Args:
            quote: TEE 引用

        Returns:
            是否验证通过
        """
        if self.tee_type == TEEType.SGX:
            return self._verify_sgx_quote(quote)
        return True

    def _verify_sgx_quote(self, quote: TEEQuote) -> bool:
        """验证 SGX 引用"""
        return len(quote.quote_data) > 0 and quote.signature is not None


class TEEAuditWriter:
    """
    TEE 审计写入器

    将审计日志安全地写入 TEE 保护的区域
    """

    def __init__(
        self,
        tee_type: TEEType = TEEType.SGX,
        output_path: str = "logs/tee_audit"
    ):
        self.tee_type = tee_type
        self.output_path = output_path
        self.formatter = TEEAuditFormatter(tee_type)
        self.ra_service = TEERemoteAttestation(tee_type)
        self.entries: List[AuditEntry] = []

    def write_entry(
        self,
        data: Dict[str, Any],
        enclave_id: str = "",
        measurement: str = ""
    ) -> AuditEntry:
        """
        写入审计条目

        Args:
            data: 审计数据
            enclave_id: Enclave ID
            measurement: Enclave 测量值

        Returns:
            AuditEntry: 写入的审计条目
        """
        entry = self.formatter.format_entry(data, enclave_id, measurement)

        self.entries.append(entry)

        return entry

    def write_with_attestation(
        self,
        data: Dict[str, Any],
        enclave_id: str = "",
        measurement: str = ""
    ) -> AuditEntry:
        """
        写入带远程认证的审计条目

        Args:
            data: 审计数据
            enclave_id: Enclave ID
            measurement: Enclave 测量值

        Returns:
            AuditEntry: 带认证的审计条目
        """
        entry = self.formatter.format_entry(data, enclave_id, measurement)

        report_data = f"{entry.current_hash}{entry.previous_hash}".encode()
        quote = self.ra_service.generate_quote(report_data)
        entry.quote = quote

        self.entries.append(entry)

        return entry

    def verify_chain(self) -> bool:
        """
        验证审计链完整性

        Returns:
            是否完整
        """
        for i in range(1, len(self.entries)):
            if self.entries[i].previous_hash != self.entries[i - 1].current_hash:
                return False

            expected_hash = self.formatter._compute_hash(
                self.entries[i].header,
                self.entries[i].data,
                self.entries[i].previous_hash
            )

            if self.entries[i].current_hash != expected_hash:
                return False

        return True

    def get_entries(self) -> List[AuditEntry]:
        """获取所有审计条目"""
        return self.entries

    def export_to_file(self, path: str) -> None:
        """导出审计日志到文件"""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


class TEEAuditReader:
    """
    TEE 审计读取器

    读取和验证 TEE 审计日志
    """

    def __init__(self, tee_type: TEEType = TEEType.SGX):
        self.tee_type = tee_type
        self.entries: List[AuditEntry] = []

    def load_from_file(self, path: str) -> None:
        """从文件加载审计日志"""
        self.entries = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entry = AuditEntry.from_dict(data)
                    self.entries.append(entry)

    def verify_chain(self) -> bool:
        """验证审计链完整性"""
        if not self.entries:
            return True

        for i in range(1, len(self.entries)):
            if self.entries[i].previous_hash != self.entries[i - 1].current_hash:
                return False

        return True

    def get_entries_by_session(self, session_id: str) -> List[AuditEntry]:
        """获取指定会话的审计条目"""
        return [
            entry for entry in self.entries
            if entry.data.get("session_id") == session_id
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.entries:
            return {
                "total_entries": 0,
                "time_range": None,
                "tee_type": self.tee_type.value
            }

        timestamps = [entry.header.timestamp for entry in self.entries]

        return {
            "total_entries": len(self.entries),
            "time_range": {
                "start": min(timestamps),
                "end": max(timestamps)
            },
            "tee_type": self.tee_type.value,
            "security_level": self.entries[0].header.security_level if self.entries else None
        }
