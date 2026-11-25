import time
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def bench(name, enc):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = smp.Unet(enc, encoder_weights=None, in_channels=3, classes=1).cuda()
    opt = torch.optim.AdamW(m.parameters(), 3e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(8, 3, 512, 512, device="cuda")
    y = torch.rand(8, 1, 512, 512, device="cuda")

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.binary_cross_entropy_with_logits(m(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(3):
        step()                      # warm-up
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    s = (time.time() - t) / 10
    train_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"{name}: {s:.3f} s/step | train peak {train_gb:.1f} GB | "
          f"4000 steps = {s * 4000 / 60:.0f} min")

    m.eval()
    for res in (1024, 1536):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            m(torch.randn(1, 3, res, res, device="cuda"))
        print(f"   inference {res}px peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    del m, opt
    torch.cuda.empty_cache()

print("GPU:", torch.cuda.get_device_name(0),
      f"| total {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
bench("ResNet-34 U-Net", "resnet34")
bench("EffNetV2-S U-Net", "tu-tf_efficientnetv2_s")