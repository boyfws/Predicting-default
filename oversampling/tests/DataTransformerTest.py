import json

import numpy as np
import pandas as pd
import pytest
import torch

from oversampling.DataTransformer import DataTransformer

test_data = pd.DataFrame(
    {
        "numeric": [1.5, np.nan, 3.0, 4.2, -1.8],  # numeric
        "categorical": ["apple", "banana", None, "apple", "cherry"],  # obj
        "binary": [True, False, True, None, False],  # obj
        "binary2": [True, False, True, True, False],  # bool
        "categorical2": ["apple", "banana", "cherry", "apple", "cherry"],  # obj
    }
)


def test_not_fitted_call1():
    with pytest.raises(RuntimeError) as e:
        transformer = DataTransformer().inverse_transform(torch.Tensor([[1, 1]]))
    assert (
        str(e.value) == "fit_transform method must be called before inverse_transform"
    )


def test_not_fitted_call2():
    with pytest.raises(RuntimeError) as e:
        transformer = DataTransformer().get_params()
    assert str(e.value) == "fit_transform method must be called before get_params"


def test_output_type():
    transformer = DataTransformer()
    transformed_tensor = transformer.fit_transform(test_data)

    assert isinstance(transformed_tensor, torch.Tensor)


def test_output_shape():
    transformer = DataTransformer()
    transformed_tensor = transformer.fit_transform(test_data)

    assert tuple(transformed_tensor.shape) == (5, 6)
    # 5 initial + 1 mask (in cat cols NaN encoded as category)
    # masks for NaN-s are used only for numeric and bool columns "binary" is an obj columns
    # as it contains NaN-s


def test_inverse_transform():
    transformer = DataTransformer()
    transformed_tensor = transformer.fit_transform(test_data)
    inverse_transformed_tensor = transformer.inverse_transform(transformed_tensor)

    pd.testing.assert_frame_equal(
        test_data,
        inverse_transformed_tensor,
        check_dtype=True,
        check_exact=False,
        atol=1e-3,
    )


def test_data_transformer_serialization_structure():
    transformer = DataTransformer()
    transformer.fit_transform(test_data)
    state = transformer.get_params()

    assert "final_scaler" in state
    assert "encoders" in state
    assert "min_max" in state
    assert "types" in state
    assert "columns" in state
    assert "apply_round" in state

    json_state = json.dumps(state, indent=2)
    assert isinstance(json_state, str)


def test_data_transformer_serialization_load():
    transformer = DataTransformer()
    transformer.fit_transform(test_data)

    state = transformer.get_params()
    loaded_transformer = DataTransformer.save_params(state)

    original_scaler = transformer.final_scaler
    loaded_scaler = loaded_transformer.final_scaler

    assert np.allclose(original_scaler.scale_, loaded_scaler.scale_)
    assert np.allclose(original_scaler.min_, loaded_scaler.min_)

    for col in transformer.encoders:
        orig_cats = transformer.encoders[col].categories_[0]
        loaded_cats = loaded_transformer.encoders[col].categories_[0]
        assert np.array_equal(orig_cats, loaded_cats)


def test_data_transformer_serialization_load_inverse():
    transformer = DataTransformer()
    transformed_tensor = transformer.fit_transform(test_data)

    state = transformer.get_params()
    loaded_transformer = DataTransformer.save_params(state)

    reconstructed_data = loaded_transformer.inverse_transform(transformed_tensor)

    pd.testing.assert_frame_equal(
        test_data, reconstructed_data, check_dtype=True, check_exact=False, atol=1e-3
    )

    assert reconstructed_data["numeric"].isna().sum() == 1
    assert reconstructed_data["categorical"].isna().sum() == 1
    assert reconstructed_data["binary"].isna().sum() == 1

    assert reconstructed_data["numeric"].dtype == np.float64
    assert pd.api.types.is_object_dtype(reconstructed_data["categorical"])
    assert pd.api.types.is_object_dtype(reconstructed_data["binary"])
    assert pd.api.types.is_bool_dtype(reconstructed_data["binary2"])


def test_edge_cases_empty_df():
    empty_data = pd.DataFrame()
    transformer = DataTransformer()
    with pytest.raises(RuntimeError) as e:
        transformer.fit_transform(empty_data)
    assert str(e.value) == "DataFrame is empty"


def test_edge_cases_singe_column():
    single_col_data = pd.DataFrame({"test": [1, 2, 3]})

    transformer = DataTransformer()
    transformed = transformer.fit_transform(single_col_data)
    state = transformer.get_params()
    loaded = DataTransformer.save_params(state)
    reconstructed = loaded.inverse_transform(transformed)

    pd.testing.assert_frame_equal(single_col_data, reconstructed, check_dtype=True)


def test_edge_cases_extreme_values():
    extreme_data = pd.DataFrame({"values": [np.nan, np.inf, -np.inf, 1e18]})
    transformer = DataTransformer()
    with pytest.raises(RuntimeError) as e:
        transformer.fit_transform(extreme_data)

    assert str(e.value) == "DataFrame is infinite"


def test_int_cols_with_na():
    data = pd.DataFrame({"col": [1, 2, 3, None]})

    transformer = DataTransformer()
    transformed = transformer.fit_transform(data)

    torch.manual_seed(42)
    transformed += torch.randn_like(transformed) * 0.02

    # Class should understand that column contains integers, except it-s data type - float64
    inverse_transformed = transformer.inverse_transform(transformed)

    pd.testing.assert_frame_equal(
        data,
        inverse_transformed,
        check_dtype=True,
        check_exact=False,
    )


def test_nan_order_cat_cols():
    # The test is written for the falling case
    # when we pass elments to OrdinalEncoder
    # in such an order that nan is not at the end
    # like that:
    # self.encoders[el] = OrdinalEncoder(
    #    categories=[df[el].unique()],
    #    encoded_missing_value=-1
    # )
    # it might give such error E

    # ValueError: Nan should be the last element in user provided categories,
    # see categories [nan 'a'] in column #0
    data = pd.DataFrame({"1": [np.nan, np.nan, np.nan, "a"]})
    transformer = DataTransformer()
    transformed = transformer.fit_transform(data)
