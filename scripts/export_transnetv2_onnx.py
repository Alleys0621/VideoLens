"""一次性脚本: 导出 TransNetV2 PyTorch 模型为 ONNX 格式

运行后生成 models/transnetv2.onnx，之后运行时只需 onnxruntime，无需 torch。
"""

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def main():
    import torch
    from transnetv2_pytorch import TransNetV2

    print("加载 TransNetV2 PyTorch 模型...")
    model = TransNetV2(device="cpu")
    model.eval()

    # 包装模型: 输入 [B, T, 27, 48, 3] uint8 → 输出 sigmoid(single_frame_pred) [B, T, 1]
    class TransNetWrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            one_hot, _ = self.base(x)
            return torch.sigmoid(one_hot)

    wrapper = TransNetWrapper(model)
    wrapper.eval()

    # 导出
    os.makedirs("models", exist_ok=True)
    output_path = "models/transnetv2.onnx"

    dummy_input = torch.zeros(1, 100, 27, 48, 3, dtype=torch.uint8)

    print(f"导出 ONNX → {output_path} ...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        opset_version=14,
        input_names=["frames"],
        output_names=["predictions"],
        dynamic_axes={
            "frames": {0: "batch"},
            "predictions": {0: "batch"},
        },
    )
    print(f"完成: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f}MB)")

    # 验证: ONNX vs PyTorch 输出一致性
    import onnxruntime

    print("\n验证 ONNX 输出...")
    session = onnxruntime.InferenceSession(output_path, providers=["CPUExecutionProvider"])

    test_input = np.random.randint(0, 256, (1, 100, 27, 48, 3), dtype=np.uint8)

    # PyTorch
    with torch.no_grad():
        pt_out = wrapper(torch.from_numpy(test_input)).numpy()

    # ONNX
    ort_out = session.run(["predictions"], {"frames": test_input})[0]

    diff = np.abs(pt_out - ort_out).max()
    print(f"  PyTorch vs ONNX 最大差异: {diff:.6f}")
    if diff < 1e-4:
        print("  验证通过")
    else:
        print("  警告: 差异较大，请检查导出参数")

    print("\nDone.")


if __name__ == "__main__":
    main()
