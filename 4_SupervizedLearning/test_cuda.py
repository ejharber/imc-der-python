import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print("PyTorch is using CUDA")
        print(f"Number of available GPU devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")
        print("PyTorch is using CPU")

if __name__ == "__main__":
    check_cuda()
