"""Offline tests; no connection to a real device is permitted."""

from unittest.mock import Mock, call

import pytest

from pymitsubishi import MitsubishiAPI, MitsubishiController
from pymitsubishi.air_purifier import HVAC_OPTIONS_REQUEST, parse_air_purifier_status


def frame(payload: bytes, packet_type: int = 0x62) -> bytes:
    """Build independently checksummed synthetic response frames."""
    body = bytes((packet_type, 1, 0x30, len(payload))) + payload
    return b"\xfc" + body + bytes((-sum(body) & 0xFF,))


def xml(*packets: bytes) -> str:
    values = "".join(f"<VALUE>{packet.hex()}</VALUE>" for packet in packets)
    return f"<LSV><CODE>{values}</CODE><MAC>not-for-service-output</MAC></LSV>"


@pytest.mark.parametrize("value", [0, 1])
def test_boolean_status(value):
    result = parse_air_purifier_status(xml(frame(bytes((0x42, value)) + bytes(14))))
    assert result.enabled is bool(value)
    assert result.response_groups == (0x42,)


def test_documented_ln25vg2w_packet():
    packet = bytes.fromhex("fc620130104201010100000000000000000000000018")
    assert parse_air_purifier_status(xml(packet)).enabled is True


@pytest.mark.parametrize("response", ["<LSV/>", xml(frame(bytes((2,)) + bytes(15))), xml(frame(b"\x00", 0x61))])
def test_missing_status_is_unknown_not_off(response):
    assert parse_air_purifier_status(response).enabled is None


@pytest.mark.parametrize("value", [2, 0xFF])
def test_unknown_boolean_values_are_rejected(value):
    with pytest.raises(ValueError, match="status value"):
        parse_air_purifier_status(xml(frame(bytes((0x42, value)) + bytes(14))))


def test_checksum_is_validated():
    packet = bytearray(frame(b"\x42\x01" + bytes(14)))
    packet[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        parse_air_purifier_status(xml(packet))


@pytest.mark.parametrize(
    "packet",
    [b"\xfc", frame(b"\x42"), frame(b"\x42\x01")[:-1], b"\x00" + frame(b"\x42\x01")[1:]],
)
def test_bad_or_incomplete_frames_are_rejected(packet):
    with pytest.raises(ValueError):
        parse_air_purifier_status(xml(packet))


@pytest.mark.parametrize("response", ["<LSV>", "<LSV><CODE><VALUE>not-hex</VALUE></CODE></LSV>"])
def test_malformed_responses_are_rejected(response):
    with pytest.raises(ValueError):
        parse_air_purifier_status(response)


def test_conflicting_responses_are_not_arbitrarily_selected():
    with pytest.raises(ValueError, match="Conflicting"):
        parse_air_purifier_status(xml(frame(b"\x42\x01"), frame(b"\x42\x00")))


def test_duplicate_identical_packets_are_accepted():
    packet = frame(b"\x42\x01" + bytes(14))
    assert parse_air_purifier_status(xml(packet, packet)).enabled is True


def test_get_request_has_no_write_flags():
    assert len(HVAC_OPTIONS_REQUEST) == 22
    assert HVAC_OPTIONS_REQUEST[:6] == bytes((0xFC, 0x42, 1, 0x30, 0x10, 0x42))
    assert HVAC_OPTIONS_REQUEST[6:-1] == bytes(15)
    assert sum(HVAC_OPTIONS_REQUEST[1:]) % 256 == 0


def test_probe_does_not_replace_normal_state_or_issue_control_calls():
    api = Mock(spec=MitsubishiAPI)
    api.send_command.return_value = xml(frame(b"\x42\x01" + bytes(14)))
    controller = MitsubishiController(api)
    cached_state = controller.state = object()
    cached_profile = controller.profile_code = [b"cached"]
    assert controller.fetch_air_purifier_status().enabled is True
    assert controller.state is cached_state
    assert controller.profile_code is cached_profile
    assert api.mock_calls == [call.send_command(HVAC_OPTIONS_REQUEST, connect=None)]


def test_fresh_missing_response_does_not_reuse_previous_success():
    api = Mock(spec=MitsubishiAPI)
    api.send_command.side_effect = [xml(frame(b"\x42\x01")), "<LSV/>"]
    controller = MitsubishiController(api)
    assert controller.fetch_air_purifier_status().enabled is True
    assert controller.fetch_air_purifier_status().enabled is None


def test_probe_leaves_cloud_and_echonet_out_of_request(monkeypatch):
    api = MitsubishiAPI("unused.invalid")
    request = Mock(return_value=xml(frame(b"\x42\x00" + bytes(14))))
    monkeypatch.setattr(api, "make_request", request)
    assert MitsubishiController(api).fetch_air_purifier_status().enabled is False
    request.assert_called_once_with(f"<CSV><CODE><VALUE>{HVAC_OPTIONS_REQUEST.hex()}</VALUE></CODE></CSV>")
    api.close()


def test_atw_probe_is_rejected_before_any_request():
    api = Mock(spec=MitsubishiAPI)
    controller = MitsubishiController(api)
    controller.is_atw = True
    with pytest.raises(ValueError, match="air-to-air"):
        controller.fetch_air_purifier_status()
    assert api.mock_calls == []


def test_transport_failure_is_not_reported_as_off():
    api = Mock(spec=MitsubishiAPI)
    api.send_command.side_effect = TimeoutError("simulated timeout")
    with pytest.raises(TimeoutError):
        MitsubishiController(api).fetch_air_purifier_status()
