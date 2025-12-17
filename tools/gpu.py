def detect_gpu():
    """Return 'cuda' if a CUDA GPU is available for PyTorch, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'
