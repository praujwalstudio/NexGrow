"""Tests for NexGrow utility modules."""

import pytest
import numpy as np
import torch


class TestMixStyle:
    def test_forward_pass(self):
        from src.utils.mixstyle import MixStyle

        mixstyle = MixStyle(p=1.0, alpha=0.1)
        mixstyle.train()
        x = torch.randn(8, 64, 56, 56)
        out = mixstyle(x)
        assert out.shape == x.shape

    def test_eval_mode_passthrough(self):
        from src.utils.mixstyle import MixStyle

        mixstyle = MixStyle(p=1.0)
        mixstyle.eval()
        x = torch.randn(4, 32, 28, 28)
        out = mixstyle(x)
        assert torch.equal(out, x)


class TestWeatherAdvisor:
    def test_treatment_recommendation(self):
        from src.utils.weather import WeatherAdvisor, WeatherData

        advisor = WeatherAdvisor()
        weather = WeatherData(
            temperature_c=30.0, humidity_pct=85.0,
            wind_speed_kmh=10.0, description="rain",
            rain_mm=5.0, location="test",
        )
        result = advisor.get_treatment("Bacterial_Blight", weather)
        assert "treatment" in result
        assert result["weather_summary"]["condition"] == "high_humidity"
