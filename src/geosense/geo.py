"""Approximate country centroids (lat, lon in degrees) for the 56 kept countries.

Used only for the great-circle distance error. Names follow data/classes.json.
Values are rounded geographic centres (not population centres): good enough for a
country-level error in km, but say "approximate" in the report.
"""
from __future__ import annotations

import numpy as np

CENTROIDS = {
    "Argentina": (-34.0, -64.0), "Australia": (-25.0, 134.0), "Austria": (47.5, 14.5),
    "Bangladesh": (23.7, 90.4), "Belgium": (50.6, 4.7), "Bolivia": (-16.7, -64.7),
    "Botswana": (-22.3, 24.7), "Brazil": (-10.8, -52.9), "Bulgaria": (42.7, 25.5),
    "Cambodia": (12.6, 104.9), "Canada": (56.1, -106.3), "Chile": (-30.0, -71.0),
    "Colombia": (4.6, -74.1), "Croatia": (45.1, 15.2), "Czechia": (49.8, 15.5),
    "Denmark": (56.0, 10.0), "Finland": (64.0, 26.0), "France": (46.6, 2.4),
    "Germany": (51.2, 10.4), "Ghana": (7.9, -1.0), "Greece": (39.1, 22.9),
    "Hungary": (47.2, 19.5), "India": (22.0, 79.0), "Indonesia": (-2.5, 118.0),
    "Ireland": (53.4, -8.2), "Israel": (31.0, 35.0), "Italy": (42.8, 12.8),
    "Japan": (36.2, 138.3), "Kenya": (0.2, 37.9), "Latvia": (56.9, 24.9),
    "Lithuania": (55.3, 23.9), "Malaysia": (4.2, 102.0), "Mexico": (23.6, -102.5),
    "Netherlands": (52.2, 5.3), "New Zealand": (-41.0, 174.0), "Nigeria": (9.1, 8.7),
    "Norway": (62.0, 10.0), "Peru": (-9.2, -75.0), "Philippines": (12.9, 122.0),
    "Poland": (52.0, 19.1), "Portugal": (39.6, -8.0), "Romania": (45.9, 25.0),
    "Russia": (61.5, 99.0), "Singapore": (1.35, 103.8), "Slovakia": (48.7, 19.7),
    "South Africa": (-29.0, 24.0), "South Korea": (36.5, 127.9), "Spain": (40.2, -3.6),
    "Sweden": (62.0, 15.0), "Switzerland": (46.8, 8.2), "Taiwan": (23.7, 121.0),
    "Thailand": (15.1, 101.0), "Turkey": (39.0, 35.2), "Ukraine": (49.0, 31.4),
    "United Kingdom": (54.0, -2.5), "United States": (39.8, -98.6),
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def distance_matrix(classes: list[str]) -> np.ndarray:
    """(C, C) matrix of centroid-to-centroid distances in km, in classes.json order."""
    ll = np.array([CENTROIDS[c] for c in classes])
    return haversine_km(ll[:, None, 0], ll[:, None, 1], ll[None, :, 0], ll[None, :, 1])
