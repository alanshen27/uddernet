"""UdderNet: neural networks for bovine mastitis detection."""

from uddernet.models import MastitisMLP, UdderCNN, get_device

__all__ = ["MastitisMLP", "UdderCNN", "get_device"]
