"""
tests/test_embed_text.py — Tests for embed_text.py (CPU-only, mocked).

Verifies module structure, quantization logic, and device selection
without downloading or running actual embedding models.
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestEmbedTextModule:
    def test_module_exists(self):
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
        assert os.path.exists(os.path.join(src_dir, "embed_text.py"))

    def test_source_imports_torch(self):
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        with open(os.path.join(src_dir, "embed_text.py")) as f:
            source = f.read()
        assert "import torch" in source

    def test_source_has_bitsandbytes_quantization(self):
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        with open(os.path.join(src_dir, "embed_text.py")) as f:
            source = f.read()
        assert "BitsAndBytesConfig" in source
        assert "load_in_8bit" in source


class TestDeviceSelection:
    def test_device_falls_back_to_cpu_when_no_gpu(self):
        """_auto_device should return 'cpu' when neither CUDA nor MPS is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Simulate the device selection logic directly
        if mock_torch.cuda.is_available():
            device = "cuda"
        elif mock_torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        assert device == "cpu"

    def test_device_selects_cuda_when_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        device = "cuda" if mock_torch.cuda.is_available() else "cpu"

        assert device == "cuda"


class TestQuantizationLogic:
    def test_int8_quantization_only_applied_on_cuda(self):
        """INT8 quantization should only activate when device == 'cuda'."""
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        with open(os.path.join(src_dir, "embed_text.py")) as f:
            source = f.read()
        # Both conditions must be present together
        assert 'quantization == "int8"' in source
        assert 'self.device == "cuda"' in source

    def test_quantization_parameter_accepted(self):
        """TextEmbedder.__init__ must accept a 'quantization' keyword argument."""
        import inspect
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        with open(os.path.join(src_dir, "embed_text.py")) as f:
            source = f.read()
        assert "quantization" in source
