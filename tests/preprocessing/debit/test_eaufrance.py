from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from previ_r2d2.preprocessing.debit.eaufrance import (
    RETRY_ATTEMPTS,
    EauFranceClient,
    series_range,
)


def _fake_response(status_ok: bool, status_code: int = 200, json_data=None):
    resp = MagicMock()
    resp.ok = status_ok
    resp.status_code = status_code
    resp.reason = "Gateway Time-out"
    resp.json.return_value = json_data or {"series": {"data": []}}
    return resp


def test_series_retries_on_transient_timeout_then_succeeds():
    client = EauFranceClient()
    ok_response = _fake_response(True, json_data={"series": {"data": [{"t": "2026-01-01", "v": 100}]}})
    failures = [requests.exceptions.ReadTimeout("timed out")] * (RETRY_ATTEMPTS - 1)

    with patch.object(client._session, "get", side_effect=[*failures, ok_response]) as mock_get, \
         patch("time.sleep") as mock_sleep:
        result = client.series("O020002001", "01/01/2026", "02/01/2026")

    assert result == {"series": {"data": [{"t": "2026-01-01", "v": 100}]}}
    assert mock_get.call_count == RETRY_ATTEMPTS
    assert mock_sleep.call_count == RETRY_ATTEMPTS - 1


def test_series_raises_after_exhausting_all_retries():
    client = EauFranceClient()

    with patch.object(client._session, "get",
                       side_effect=requests.exceptions.ReadTimeout("timed out")) as mock_get, \
         patch("time.sleep") as mock_sleep:
        with pytest.raises(requests.exceptions.ReadTimeout):
            client.series("O020002001", "01/01/2026", "02/01/2026")

    assert mock_get.call_count == RETRY_ATTEMPTS
    assert mock_sleep.call_count == RETRY_ATTEMPTS - 1


def test_series_retries_on_http_504_then_succeeds():
    client = EauFranceClient()
    bad_response = _fake_response(False, status_code=504)
    ok_response = _fake_response(True, json_data={"series": {"data": [{"t": "2026-01-01", "v": 50}]}})

    with patch.object(client._session, "get", side_effect=[bad_response, ok_response]) as mock_get, \
         patch("time.sleep") as mock_sleep:
        result = client.series("O020002001", "01/01/2026", "02/01/2026")

    assert result == {"series": {"data": [{"t": "2026-01-01", "v": 50}]}}
    assert mock_get.call_count == 2
    assert mock_sleep.call_count == 1


def test_series_does_not_retry_on_success():
    client = EauFranceClient()
    ok_response = _fake_response(True, json_data={"series": {"data": []}})

    with patch.object(client._session, "get", return_value=ok_response) as mock_get, \
         patch("time.sleep") as mock_sleep:
        client.series("O020002001", "01/01/2026", "02/01/2026")

    assert mock_get.call_count == 1
    assert mock_sleep.call_count == 0


def test_series_range_uses_single_call_for_short_range():
    client = MagicMock()
    client.series.return_value = {"series": {"data": [{"t": "2026-01-01", "v": 10}]}}

    result = series_range(client, "O020002001", "01/01/2026", "10/01/2026", chunk_days=180)

    client.series.assert_called_once_with("O020002001", "01/01/2026", "10/01/2026")
    assert result == {"series": {"data": [{"t": "2026-01-01", "v": 10}]}}


def test_series_range_splits_long_range_into_contiguous_non_overlapping_chunks():
    client = MagicMock()
    client.series.return_value = {"series": {"data": []}}

    series_range(client, "O020002001", "01/01/2021", "01/01/2022", chunk_days=180)

    calls = [c.args for c in client.series.call_args_list]
    assert calls == [
        ("O020002001", "01/01/2021", "29/06/2021"),
        ("O020002001", "30/06/2021", "26/12/2021"),
        ("O020002001", "27/12/2021", "01/01/2022"),
    ]


def test_series_range_concatenates_data_from_all_chunks_in_order():
    client = MagicMock()
    client.series.side_effect = [
        {"series": {"data": [{"t": "2021-01-01", "v": 1}]}},
        {"series": {"data": [{"t": "2021-07-01", "v": 2}]}},
    ]

    result = series_range(client, "O020002001", "01/01/2021", "01/09/2021", chunk_days=180)

    assert result == {
        "series": {"data": [{"t": "2021-01-01", "v": 1}, {"t": "2021-07-01", "v": 2}]}
    }
