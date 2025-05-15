import numpy as np
import pandas as pd
import pytest
import torch

from oversampling.VAE import VAEWrapper

test_data = pd.DataFrame(
    {
        "numeric": [1.5, np.nan, 3.0, 4.2, -1.8],  # numeric
        "categorical": ["apple", "banana", None, "apple", "cherry"],  # obj
        "binary": [True, False, True, None, False],  # obj
        "binary2": [True, False, True, True, False],  # bool
        "categorical2": ["apple", "banana", "cherry", "apple", "cherry"],  # obj
    }
)


def test_not_fitted_call2():
    model = VAEWrapper()
    with pytest.raises(RuntimeError) as e:
        model.save("path.json")
    assert str(e.value) == "Instance has not been fitted yet."


def test_not_fitted_call3():
    model = VAEWrapper()
    with pytest.raises(RuntimeError) as e:
        model.plot_fit((10, 10))
    assert str(e.value) == "Instance has not been fitted yet."


def test_random_state():
    PARAMS = dict(
        latent_dim=1,
        dims=(2, 3, 4, 5),
        lr=1e-4,
        batch_size=2,
        epochs=1,
        seed=42,
        leaky_relu_coef=0.2,
    )

    model1 = VAEWrapper(**PARAMS)
    model1.fit(test_data)
    gen1 = model1.to_latent(test_data, seed=42)

    model2 = VAEWrapper(**PARAMS)
    model2.fit(test_data)
    gen2 = model2.to_latent(test_data, seed=42)

    PARAMS["seed"] = 21
    model3 = VAEWrapper(**PARAMS)
    model3.fit(test_data)
    gen3 = model3.to_latent(test_data, seed=42)

    assert torch.allclose(gen1, gen2)
    # same seed, they should be same

    # different seeds, they should be different
    assert not torch.allclose(gen1, gen3)


def test_random_state_in_generation():
    PARAMS = dict(
        latent_dim=1,
        dims=(2, 3, 4, 5),
        lr=1e-4,
        batch_size=2,
        epochs=1,
        seed=42,
        leaky_relu_coef=0.2,
    )

    model = VAEWrapper(**PARAMS)
    model.fit(test_data)
    gen1 = model.to_latent(test_data, seed=42)
    gen2 = model.to_latent(test_data, seed=42)

    assert torch.allclose(gen1, gen2)


def test_model_saving(tmp_path):
    target_path = tmp_path / "G.pth"

    PARAMS = dict(
        latent_dim=1,
        dims=(2, 3, 4, 5),
        lr=1e-4,
        batch_size=2,
        epochs=1,
        seed=42,
        leaky_relu_coef=0.2,
    )

    model = VAEWrapper(**PARAMS)
    model.fit(test_data)
    gen = model.to_latent(test_data, seed=42)

    model.save(target_path)

    loaded_model = VAEWrapper(**PARAMS).load(target_path)

    assert loaded_model.latent_dim == PARAMS["latent_dim"]
    assert loaded_model.dims == PARAMS["dims"]
    assert loaded_model.input_dim == model.input_dim

    gen_loaded = loaded_model.to_latent(test_data, seed=42)

    assert torch.allclose(gen, gen_loaded)


def test_dropout():
    PARAMS = dict(
        latent_dim=1,
        dims=(2, 3, 4, 5),
        lr=1e-4,
        batch_size=2,
        epochs=1,
        seed=42,
        leaky_relu_coef=0.2,
        dropout=(0.33, 0.5),
    )

    model = VAEWrapper(**PARAMS)
    model.fit(test_data)
    gen = model.to_latent(test_data, seed=42)
