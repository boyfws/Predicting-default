import pytest
import pandas as pd
import numpy as np

from oversampling import OversampleGAN

test_data = pd.DataFrame({
    'numeric': [1.5, np.nan, 3.0, 4.2, -1.8],  # numeric
    'categorical': ['apple', 'banana', None, 'apple', 'cherry'],  # obj
    'binary': [True, False, True, None, False],  # obj
    "binary2": [True, False, True, True, False],  # bool
    'categorical2': ['apple', 'banana', "cherry", 'apple', 'cherry'],  # obj
})


def test_not_fitted_call1():
    model = OversampleGAN()
    with pytest.raises(RuntimeError) as e:
        model.generate(100)
    assert str(e.value) == "Instance has not been fitted yet."


def test_not_fitted_call2():
    model = OversampleGAN()
    with pytest.raises(RuntimeError) as e:
        model.save_generator("path.json")
    assert str(e.value) == "Instance has not been fitted yet."


def test_not_fitted_call3():
    model = OversampleGAN()
    with pytest.raises(RuntimeError) as e:
        model.plot_fit((10, 10))
    assert str(e.value) == "Instance has not been fitted yet."


def test_random_state():
    PARAMS = dict(
        latent_dim=1,
        hidden_dims=(2, 3),
        D_lr=1e-4,
        G_lr=4e-4,
        batch_size=2,
        epochs=1,
        seed=42,
        leaky_relu_coef=0.2,
    )

    model1 = OversampleGAN(**PARAMS)
    model1.fit(test_data)
    gen1 = model1.generate(100)

    model2 = OversampleGAN(**PARAMS)
    model2.fit(test_data)
    gen2 = model2.generate(100)

    PARAMS["seed"] = 21
    model3 = OversampleGAN(**PARAMS)
    model3.fit(test_data)
    gen3 = model3.generate(100)

    # same seed, they should be same
    pd.testing.assert_frame_equal(
        gen1,
        gen2,
        check_dtype=True,
        check_exact=False,
        atol=1e-3
    )

    # different seeds, they should be different
    with pytest.raises(AssertionError) as e:
        pd.testing.assert_frame_equal(
            gen1,
            gen3,
            check_dtype=True,
            check_exact=False,
            atol=1e-3
        )


def test_random_state_in_generation():
    PARAMS = dict(
        latent_dim=1,
        hidden_dims=(2, 3),
        D_lr=1e-4,
        G_lr=4e-4,
        batch_size=2,
        epochs=1,
        seed=42,
        leaky_relu_coef=0.2,
    )

    model = OversampleGAN(**PARAMS)
    model.fit(test_data)
    gen1 = model.generate(100, seed=42)
    gen2 = model.generate(100, seed=42)

    pd.testing.assert_frame_equal(
        gen1,
        gen2,
        check_dtype=True,
    )


def test_model_saving(tmp_path):
    target_path = tmp_path / "G.pth"

    PARAMS = dict(
        latent_dim=1,
        hidden_dims=(2, 3),
        D_lr=1e-4,
        G_lr=4e-4,
        batch_size=2,
        epochs=1,
        seed=42,
        leaky_relu_coef=0.2,
    )

    model = OversampleGAN(**PARAMS)
    model.fit(test_data)
    gen = model.generate(100, seed=42)

    model.save_generator(target_path)

    loaded_model = OversampleGAN(**PARAMS).load_generator(target_path)

    assert loaded_model.latent_dim == PARAMS["latent_dim"]
    assert loaded_model.hidden_dims == PARAMS["hidden_dims"]
    assert loaded_model.input_dim == model.input_dim

    gen_loaded = loaded_model.generate(100, seed=42)

    pd.testing.assert_frame_equal(
        gen,
        gen_loaded,
        check_dtype=True,
    )


def test_dropout():
    PARAMS = dict(
        latent_dim=1,
        hidden_dims=(2, 3),
        D_lr=1e-4,
        G_lr=4e-4,
        batch_size=2,
        epochs=1,
        seed=42,
        leaky_relu_coef=0.2,
        dropout=(0.33, 0.5),
    )

    model = OversampleGAN(**PARAMS)
    model.fit(test_data)
    gen = model.generate(100, seed=42)



