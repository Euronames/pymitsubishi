"""Experimental read-only probe for CN105 HVAC options through the Wi-Fi adapter.

Protocol reference: https://muart-group.github.io/developer/it-protocol/
0x62-get-response/0x42-unknown-hvac-options

Absence of this packet is not proof that the indoor unit lacks a purifier.
"""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

# Get request (0x42), requesting HVAC options (also 0x42), with no set flags.
HVAC_OPTIONS_REQUEST = bytes.fromhex("fc42013010420000000000000000000000000000003b")


@dataclass(frozen=True)
class AirPurifierStatus:
    """A fresh observation, separate from the controller's normal cached state."""

    enabled: bool | None
    response_groups: tuple[int, ...]


def parse_air_purifier_status(response: str) -> AirPurifierStatus:
    """Decode only validated purifier status; never treat a missing value as off.

    Return group identifiers, not the XML wrapper, which may contain identity or
    other sensitive information. Malformed/conflicting packets raise ValueError.
    """
    try:
        root = ET.fromstring(response)
    except ET.ParseError as err:
        raise ValueError("Invalid HVAC options XML response") from err

    groups: set[int] = set()
    readings: set[bool] = set()
    for element in root.findall(".//CODE/VALUE"):
        if not element.text or not element.text.strip():
            continue
        try:
            packet = bytes.fromhex(element.text)
        except ValueError as err:
            raise ValueError("Invalid HVAC options packet encoding") from err
        if (
            len(packet) < 6
            or packet[0] != 0xFC
            or packet[2:4] != b"\x01\x30"
            or len(packet) != packet[4] + 6
            or sum(packet[1:]) % 256 != 0
        ):
            raise ValueError("Invalid HVAC options packet framing or checksum")
        if packet[1] != 0x62 or packet[4] == 0:
            continue
        groups.add(packet[5])
        if packet[5] != 0x42:
            continue
        if packet[4] < 2 or packet[6] not in (0, 1):
            raise ValueError("Invalid air purifier status value")
        readings.add(bool(packet[6]))

    if len(readings) > 1:
        raise ValueError("Conflicting air purifier status values")
    return AirPurifierStatus(
        enabled=next(iter(readings)) if readings else None,
        response_groups=tuple(sorted(groups)),
    )
