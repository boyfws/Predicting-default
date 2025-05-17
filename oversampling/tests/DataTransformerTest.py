import json

import numpy as np
import pandas as pd
import pytest
import torch

from oversampling.DataTransformer import DataTransformer, MISSING_VALUE

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
        str(e.value) == "fit method must be called before inverse_transform"
    )


def test_not_fitted_call2():
    with pytest.raises(RuntimeError) as e:
        transformer = DataTransformer().get_params()
    assert str(e.value) == "fit method must be called before get_params"


def test_output_type():
    transformer = DataTransformer()
    transformed_tensor = transformer.fit_transform(test_data)

    assert isinstance(transformed_tensor, torch.Tensor)



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


def test_data_transformer_serialization_load():
    transformer = DataTransformer()
    transformer.fit_transform(test_data)

    state = transformer.get_params()
    loaded_transformer = DataTransformer.save_params(state)

    original_scaler = transformer._final_scaler
    loaded_scaler = loaded_transformer._final_scaler

    assert np.allclose(original_scaler.scale_, loaded_scaler.scale_)
    assert np.allclose(original_scaler.min_, loaded_scaler.min_)



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


# Fixtures for sample data
@pytest.fixture
def simple_numeric_df():
    return pd.DataFrame({
        'a': [1.0, 2.0, np.nan, 4.0],
        'b': [0, 1, 0, 1]
    })

@pytest.fixture
def simple_categorical_df():
    return pd.DataFrame({
        'cat': ['x', 'y', None, 'x'],
        'bin': [True, False, True, False]
    })

# 1. Test output dimensions and range
def test_transform_shape_and_range(simple_numeric_df):
    transformer = DataTransformer()
    tensor = transformer.fit_transform(simple_numeric_df)
    assert isinstance(tensor, torch.Tensor)
    rows, cols = tensor.shape
    assert rows == simple_numeric_df.shape[0]
    assert cols == transformer._final_size
    assert torch.all(tensor <= 1.0) and torch.all(tensor >= -1.0)

# 2. Test unseen category error
def test_transform_unseen_category(simple_categorical_df):
    transformer = DataTransformer()
    transformer.fit(simple_categorical_df)
    new_df = simple_categorical_df.copy()
    new_df.loc[0, 'cat'] = 'z'  # unseen
    with pytest.raises(ValueError) as e:
        transformer.transform(new_df)
    assert 'Unexpected categories' in str(e.value)

# 3. Test mask creation for all-NaN numeric
def test_all_na_numeric_column():
    df = pd.DataFrame({'x': [np.nan, np.nan]})
    transformer = DataTransformer()
    tensor = transformer.fit_transform(df)
    inv = transformer.inverse_transform(tensor)
    assert inv['x'].isna().all()

# 4. Test all-NaN categorical column
def test_all_na_categorical_column():
    df = pd.DataFrame({'c': [None, None, None]})
    transformer = DataTransformer()
    tensor = transformer.fit_transform(df)
    inv = transformer.inverse_transform(tensor)
    assert inv['c'].isna().all()

# 5. Test apply_round for integer-like floats
def test_apply_round_flag():
    df = pd.DataFrame({'i': [1.0, 2.0, 3.0]})
    transformer = DataTransformer()
    t = transformer.fit_transform(df)
    inv = transformer.inverse_transform(t)
    assert inv['i'].dtype == df['i'].dtype
    assert (inv['i'] == df['i']).all()

# 6. Test calculate_mse_vae simple case
def test_calculate_mse_vae_simple():
    df = pd.DataFrame({'num': [1.0, 2.0]})
    transformer = DataTransformer()
    t = transformer.fit_transform(df)
    rec = t + 1.0
    loss = transformer.calculate_mse_vae(rec, t)
    # loss = sum((t+1 - t)^2) = rows * 1^2
    assert loss.item() == df.shape[0]

# 7. Test calculate_mse_vae masked case
def test_calculate_mse_vae_masked():
    df = pd.DataFrame({'m': [1.0, None]})
    transformer = DataTransformer()
    t = transformer.fit_transform(df)
    # create rec and true with known values
    rec = torch.zeros_like(t)
    # true = t
    true = t.clone()
    loss = transformer.calculate_mse_vae(rec, true)
    # compute expected: mask second row masked idx = 1st feature and mask column
    # mask index last column
    fi = transformer._masked_idx[(transformer._columns_data['m']['col_idx'], transformer._columns_data['m']['mask_idx'])]
    # just check loss is positive
    assert loss.item() > 0

# 8. Test get_params keys
def test_get_params_contents(simple_categorical_df):
    transformer = DataTransformer()
    transformer.fit(simple_categorical_df)
    params = transformer.get_params()
    for key in ['columns_data', 'initial_columns_order', 'final_size', 'final_scaler']:
        assert key in params

# 9. Test fit twice resets state
def test_fit_twice_simple(simple_numeric_df):
    transformer = DataTransformer()
    transformer.fit(simple_numeric_df)
    size1 = transformer._final_size
    transformer.fit(pd.DataFrame({'x': [1,2,3]}))
    size2 = transformer._final_size
    assert size2 != size1

# 10. Test saving and loading params consistency
def test_save_load_consistency(simple_categorical_df):
    transformer = DataTransformer()
    t = transformer.fit_transform(simple_categorical_df)
    params = transformer.get_params()
    loaded = DataTransformer.save_params(params)
    rec = loaded.inverse_transform(t)
    pd.testing.assert_frame_equal(rec, simple_categorical_df, check_dtype=False)

# 11. Test unexpected infinite values
def test_infinite_values():
    df = pd.DataFrame({'v': [np.inf, -np.inf, 1]})
    transformer = DataTransformer()
    with pytest.raises(RuntimeError):
        transformer.fit_transform(df)

# 12. Test single column case
def test_single_column_roundtrip():
    df = pd.DataFrame({'o': [5, 6, 7]})
    transformer = DataTransformer()
    t = transformer.fit_transform(df)
    inv = transformer.inverse_transform(t)
    pd.testing.assert_frame_equal(inv, df)

# 13. Test transform without fit
def test_transform_without_fit_error():
    transformer = DataTransformer()
    with pytest.raises(RuntimeError):
        transformer.transform(pd.DataFrame({'a':[1]}))

# 14. Test get_params without fit
def test_get_params_without_fit_error():
    transformer = DataTransformer()
    with pytest.raises(RuntimeError):
        transformer.get_params()

# 15. Test final_size matches transform output cols
def test_final_size_matches_columns(simple_numeric_df):
    transformer = DataTransformer()
    transformer.fit(simple_numeric_df)
    assert transformer._final_size == transformer.transform(simple_numeric_df).shape[1]

# 16. Test order preservation
def test_column_order_preserved(simple_categorical_df):
    df = simple_categorical_df.copy()[['bin','cat']]
    transformer = DataTransformer()
    t = transformer.fit_transform(df)
    inv = transformer.inverse_transform(t)
    assert list(inv.columns) == ['bin','cat']

# 17. Test dtype preservation for bool column
def test_bool_dtype_preserved():
    df = pd.DataFrame({'b': pd.Series([True,False,True], dtype='bool')})
    transformer = DataTransformer()
    t = transformer.fit_transform(df)
    inv = transformer.inverse_transform(t)
    assert pd.api.types.is_bool_dtype(inv['b'])

# 18. Test transformer with no numeric columns
def test_only_categorical():
    df = pd.DataFrame({'c1':['a','b'], 'c2':['x',None]})
    transformer = DataTransformer()
    t = transformer.fit_transform(df)
    inv = transformer.inverse_transform(t)
    pd.testing.assert_frame_equal(inv, df)

# 19. Test MISSING_VALUE constant used
def test_missing_value_constant():
    df = pd.DataFrame({'cat':[None,'a']})
    transformer = DataTransformer()
    transformer.fit(df)
    cats = transformer._columns_data['cat']['one_hot']['categories']
    assert MISSING_VALUE in cats

# 20. Test transform on different dataframe shape
def test_transform_different_dataframe_shape(simple_numeric_df):
    transformer = DataTransformer()
    transformer.fit(simple_numeric_df)
    new_df = simple_numeric_df.head(2)
    t = transformer.transform(new_df)
    assert t.shape[0] == 2
