import torch

from models.model_factory import get_model


def test_model(model_name, x):
    print(f"\nTesting: {model_name}")

    model = get_model(model_name)
    model = model.to("cuda")
    model.train()

    x = x.clone().detach().to("cuda")
    x.requires_grad_(True)

    output = model(x)

    print(f"  Input shape : {tuple(x.shape)}")
    print(f"  Output shape: {tuple(output.shape)}")
    print(f"  Output device: {output.device}")

    assert output.shape == (x.size(0), 3), \
        f"Expected {(x.size(0), 3)}, got {tuple(output.shape)}"

    loss = output.mean()
    loss.backward()

    print("  Forward pass: OK")
    print("  Backward pass: OK")

    # Check that the VQC parameters received gradients for PHN models
    if model_name == "phn":
        vqc = model.phn_vqc_branch.vqc_branch

        assert vqc.weights.grad is not None, \
            "VQC weights did not receive gradients."

        assert vqc.projection.weight.grad is not None, \
            "VQC projection did not receive gradients."

        print("  VQC gradients: OK")

    print(f"  {model_name}: PASSED")


def main():
    assert torch.cuda.is_available(), "CUDA is not available."

    print("CUDA device:", torch.cuda.get_device_name(0))

    # IMPORTANT: real config has 7 input features.
    # Input = (batch, lookback, NUM_FEATURES)
    x = torch.randn(4, 24, 7)

    # Verify the new PHN.
    test_model("phn", x)

    print("\nSmoke test completed successfully.")


if __name__ == "__main__":
    main()