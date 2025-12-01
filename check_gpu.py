import sys
import platform

def check_pytorch():
    print("-" * 30)
    print("Checking PyTorch...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # 1. CUDA (NVIDIA)
        if torch.cuda.is_available():
            print(f"✅ CUDA is available.")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("❌ CUDA is not available.")

        # 2. MPS (Apple Silicon / Metal)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(f"✅ MPS (Metal Performance Shaders) is available (macOS).")
        else:
            print("❌ MPS is not available.")

        # 3. HIP (AMD ROCm) - often masquerades as CUDA, but we can check version
        if hasattr(torch.version, "hip") and torch.version.hip:
            print(f"ℹ️  ROCm (HIP) version: {torch.version.hip}")
        
        # 4. XPU (Intel) - Newer PyTorch versions
        if hasattr(torch, "xpu") and torch.xpu.is_available():
             print(f"✅ Intel XPU is available.")
             print(f"   Device count: {torch.xpu.device_count()}")
             print(f"   Current device: {torch.xpu.get_device_name(0)}")

        # 5. DirectML (Windows - AMD/Intel/NVIDIA)
        try:
            import torch_directml
            if torch_directml.is_available():
                print(f"✅ DirectML is available.")
                print(f"   Device count: {torch_directml.device_count()}")
                print(f"   Current device: {torch_directml.device_name(0)}")
            else:
                print("❌ DirectML is installed but not available.")
        except ImportError:
            print("❌ DirectML (torch-directml) is not installed.")

    except ImportError:
        print("❌ PyTorch is not installed.")
    except Exception as e:
        print(f"⚠️  Error checking PyTorch: {e}")

def check_tensorflow():
    print("-" * 30)
    print("Checking TensorFlow...")
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow detected {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"   - {gpu}")
        else:
            print("❌ TensorFlow did not detect any GPUs.")
            if platform.system() == "Windows":
                try:
                    from packaging import version
                    if version.parse(tf.__version__) >= version.parse("2.11.0"):
                        print("\n⚠️  NOTE: TensorFlow 2.11+ does not support GPUs on Windows Native.")
                        print("   To use GPU with TensorFlow on Windows, you must use WSL2 (Windows Subsystem for Linux).")
                        print("   Alternatively, use PyTorch (which supports Windows Native GPU) or an older TensorFlow version (<=2.10).")
                except ImportError:
                    # Fallback if packaging is not installed
                    major, minor, *_ = tf.__version__.split('.')
                    if int(major) > 2 or (int(major) == 2 and int(minor) >= 11):
                        print("\n⚠️  NOTE: TensorFlow 2.11+ does not support GPUs on Windows Native.")
                        print("   To use GPU with TensorFlow on Windows, you must use WSL2 (Windows Subsystem for Linux).")
            
        # Check for other devices like TPU
        tpus = tf.config.list_physical_devices('TPU')
        if tpus:
            print(f"✅ TensorFlow detected {len(tpus)} TPU(s).")

    except ImportError:
        print("❌ TensorFlow is not installed.")
    except Exception as e:
        print(f"⚠️  Error checking TensorFlow: {e}")

def check_system_info():
    print("=" * 30)
    print("System Information")
    print("=" * 30)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")

if __name__ == "__main__":
    check_system_info()
    check_pytorch()
    check_tensorflow()
    print("=" * 30)
