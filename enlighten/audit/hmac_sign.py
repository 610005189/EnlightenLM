"""
HMAC Signature - HMAC签名模块
用于审计条目的身份认证和完整性验证
"""

import hmac
import hashlib
import secrets
import time
from typing import Optional, Dict, Any


class HMACSigner:
    """
    HMAC签名器

    功能:
    - 生成HMAC签名
    - 支持密钥轮换
    - 时间戳验证
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "sha256",
        key_rotation_interval: int = 86400
    ):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.algorithm = algorithm
        self.key_rotation_interval = key_rotation_interval
        self.creation_time = time.time()

        self.hash_func = getattr(hashlib, algorithm)

    def sign(self, data: Dict[str, Any], timestamp: Optional[float] = None) -> str:
        """
        生成HMAC签名

        Args:
            data: 要签名的数据
            timestamp: 可选的时间戳

        Returns:
            signature: HMAC签名
        """
        timestamp = timestamp or time.time()

        message = self._prepare_message(data, timestamp)

        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            self.hash_func
        ).hexdigest()

        return signature

    def _prepare_message(self, data: Dict, timestamp: float) -> str:
        """
        准备签名消息
        """
        data_str = str(sorted(data.items()))
        message = f"{data_str}:{timestamp}"
        return message

    def verify(
        self,
        data: Dict[str, Any],
        signature: str,
        timestamp: Optional[float] = None,
        tolerance: float = 300
    ) -> bool:
        """
        验证HMAC签名

        Args:
            data: 原始数据
            signature: 要验证的签名
            timestamp: 时间戳
            tolerance: 时间容差（秒）

        Returns:
            valid: 签名是否有效
        """
        timestamp = timestamp or time.time()

        current_time = time.time()
        if abs(current_time - timestamp) > tolerance:
            return False

        expected_signature = self.sign(data, timestamp)

        return hmac.compare_digest(signature, expected_signature)

    def should_rotate_key(self) -> bool:
        """
        检查是否需要轮换密钥
        """
        return (time.time() - self.creation_time) > self.key_rotation_interval

    def rotate_key(self) -> str:
        """
        轮换密钥

        Returns:
            new_key: 新密钥
        """
        old_key = self.secret_key
        self.secret_key = secrets.token_hex(32)
        self.creation_time = time.time()
        return old_key


class HMACVerifier:
    """
    HMAC验证器

    支持多个签名器（用于密钥轮换期间验证旧签名）
    """

    def __init__(self):
        self.signers = {}

    def add_signer(self, key_id: str, signer: HMACSigner) -> None:
        """
        添加签名器

        Args:
            key_id: 密钥ID
            signer: HMAC签名器
        """
        self.signers[key_id] = signer

    def sign(
        self,
        key_id: str,
        data: Dict[str, Any],
        timestamp: Optional[float] = None
    ) -> Dict[str, str]:
        """
        使用指定密钥签名

        Args:
            key_id: 密钥ID
            data: 要签名的数据
            timestamp: 时间戳

        Returns:
            dict包含signature和key_id
        """
        if key_id not in self.signers:
            raise ValueError(f"Unknown key_id: {key_id}")

        signer = self.signers[key_id]
        signature = signer.sign(data, timestamp)

        return {
            "signature": signature,
            "key_id": key_id,
            "timestamp": timestamp or time.time()
        }

    def verify(
        self,
        data: Dict[str, Any],
        signature: str,
        key_id: str,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        验证签名

        Args:
            data: 原始数据
            signature: 签名
            key_id: 密钥ID
            timestamp: 时间戳

        Returns:
            valid: 签名是否有效
        """
        if key_id not in self.signers:
            return False

        signer = self.signers[key_id]
        return signer.verify(data, signature, timestamp)


class StatelessHMAC:
    """
    无状态HMAC

    不存储密钥，只用于验证
    """

    def __init__(self, secret_key: str, algorithm: str = "sha256"):
        self.secret_key = secret_key
        self.hash_func = getattr(hashlib, algorithm)

    def create_token(
        self,
        data: Dict[str, Any],
        expiration: Optional[int] = None
    ) -> str:
        """
        创建带过期时间的token

        Args:
            data: 数据
            expiration: 过期时间（秒）

        Returns:
            token: 完整token
        """
        import base64
        import json

        payload = data.copy()
        if expiration:
            payload["exp"] = time.time() + expiration

        payload_bytes = json.dumps(payload, sort_keys=True, default=str).encode()
        payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode()

        signature = hmac.new(
            self.secret_key.encode(),
            payload_b64.encode(),
            self.hash_func
        ).hexdigest()

        token = f"{payload_b64}.{signature}"
        return token

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        验证token

        Args:
            token: 要验证的token

        Returns:
            data: 解压后的数据，如果无效则返回None
        """
        import base64
        import json

        try:
            payload_b64, signature = token.split(".")
            expected_signature = hmac.new(
                self.secret_key.encode(),
                payload_b64.encode(),
                self.hash_func
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                return None

            payload_bytes = base64.urlsafe_b64decode(payload_b64.encode())
            payload = json.loads(payload_bytes)

            if "exp" in payload:
                if time.time() > payload["exp"]:
                    return None

            return payload

        except Exception:
            return None
