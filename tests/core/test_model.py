from core.model import get_model_features
from core.consts import (
    TIME_FEATURES, LAG_FEATURES, WEATHER_FEATURES, HOLIDAY_FEATURES
)
from core.types import ModelFeatureFlags
# from IPython.core.debugger import set_trace


class TestModel:
    def test_get_model_features(self):
        t_set = set(TIME_FEATURES)
        l_set = set(LAG_FEATURES)
        w_set = set(WEATHER_FEATURES)
        h_set = set(HOLIDAY_FEATURES)

        base_features = set(get_model_features())
        assert t_set.issubset(base_features)
        assert not l_set.issubset(base_features)
        assert not w_set.issubset(base_features)
        assert not h_set.issubset(base_features)

        lag_features = set(get_model_features(ModelFeatureFlags(lag=True)))
        assert t_set.issubset(lag_features)
        assert l_set.issubset(lag_features)
        assert not w_set.issubset(lag_features)
        assert not h_set.issubset(lag_features)

        lw_features = set(get_model_features(ModelFeatureFlags(lag=True, weather=True)))
        assert t_set.issubset(lw_features)
        assert l_set.issubset(lw_features)
        assert w_set.issubset(lw_features)
        assert not h_set.issubset(lw_features)

        lwh_features = set(get_model_features(
            ModelFeatureFlags(lag=True, weather=True, holidays=True))
        )
        assert t_set.issubset(lwh_features)
        assert l_set.issubset(lwh_features)
        assert w_set.issubset(lwh_features)
        assert h_set.issubset(lwh_features)

        h_features = set(get_model_features(ModelFeatureFlags(holidays=True)))
        assert t_set.issubset(h_features)
        assert not l_set.issubset(h_features)
        assert not w_set.issubset(h_features)
        assert h_set.issubset(h_features)
