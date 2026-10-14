"""Candidate solution for the assignment "Map, Data Association & Path Planning".

Run it with::

    python3 solution.py

The script executes the full pipeline step by step and stops at the first
step you have not implemented yet, with a reminder of what is missing.
Implement the three functions below (``associate``, ``correct_map``,
``find_path``), keeping their names and signatures unchanged.
"""

import math
from typing import List, Tuple

from map_generation import Map, Point, generate_map, generate_observations
from visualization import (
    visualize_association_n_correction,
    visualize_map_n_observations,
    visualize_path,
)

# ---------------------------------------------------------------------------
# Parameters of the run.
#
# Feel free to experiment with them, but note that your solution will also be
# evaluated with different values (especially with different random seeds).
# ---------------------------------------------------------------------------
NUM_POINTS = 20
X_RANGE = 20.0
Y_RANGE = 20.0
MAP_SEED = 42

NOISE_STD = 0.25
OBSERVATION_RATE = 0.8
OBSERVATIONS_SEED = 7


def segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orientation(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - \
            (q[1] - p[1]) * (r[0] - p[0])

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    return o1 * o2 < 0 and o3 * o4 < 0


# ---------------------------------------------------------------------------
# Provided helper
# ---------------------------------------------------------------------------

def path_length(points: List[Point], path: List[int]) -> float:
    """Total Euclidean length of ``path`` over ``points``.

    Useful to measure how short your path is, e.g. when going for Bonus 1.
    The length is the sum of the Euclidean distances between the consecutive
    nodes of the path.
    """
    return sum(math.dist(points[a], points[b]) for a, b in zip(path, path[1:]))


# ---------------------------------------------------------------------------
# Step 3 -- Data association (yours to implement)
# ---------------------------------------------------------------------------

"""Associate every observation with the map point it was generated from.

    Given the map and the observations (noisy, incomplete and shuffled),
    recover the observation-to-map-point correspondence.

    Returns:
        The association couples as a list of ``(map_index, observation_index)``
        tuples, e.g. ``[(0, 12), (3, 5)]`` means that observation 12 was
        generated from map point 0, and observation 5 from map point 3.
    """


def associate(map_points: Map, observations: List[Point]) -> List[Tuple[int, int]]:
    associations: List[Tuple[int, int]] = []

    for i, point in enumerate(map_points):
        candidate_obs = observations[0]
        candidate_idx = 0
        for j, obs in enumerate(observations[1:], start=1):
            if math.dist(obs, point) < math.dist(candidate_obs, point):
                candidate_obs = obs
                candidate_idx = j

        associations.append((i, candidate_idx))

    for i in range(len(associations)):
        for j in range(i + 1, len(associations)):
            if associations[i][1] == associations[j][1] and associations[i][1] != -1:
                obs_idx = associations[i][1]

                dist_i = math.dist(
                    map_points[associations[i][0]],
                    observations[obs_idx]
                )
                dist_j = math.dist(
                    map_points[associations[j][0]],
                    observations[obs_idx]
                )

                if dist_i < dist_j:
                    associations[j] = (associations[j][0], -1)
                else:
                    associations[i] = (associations[i][0], -1)

    associations = [pair for pair in associations if pair[1] != -1]
    return associations


# ---------------------------------------------------------------------------
# Step 4 -- Map correction (yours to implement)
# ---------------------------------------------------------------------------

def correct_map(
        map_points: Map,
        observations: List[Point],
        associations: List[Tuple[int, int]],
) -> Map:
    """Correct the map, given the data association.

    For every association couple ``(i, j)``, the corrected position of map
    point ``i`` is the average (midpoint) of ``map_points[i]`` and
    ``observations[j]``. Map points without an associated observation keep
    their original position.

    Returns:
        The corrected map: a list of ``(x, y)`` tuples with the same length
        and the same order as ``map_points``.
    """

    corrected_map = map_points.copy()

    for map_idx, obs_idx in associations:
        x_map, y_map = map_points[map_idx]
        x_obs, y_obs = observations[obs_idx]

        corrected_map[map_idx] = (
            (x_map + x_obs) / 2,
            (y_map + y_obs) / 2
        )

    return corrected_map


# ---------------------------------------------------------------------------
# Step 5 -- Path finding (yours to implement)
# ---------------------------------------------------------------------------

def find_path(corrected_map: Map) -> List[int]:
    n = len(corrected_map)
    if n == 0:
        return []

    path = [0]

    unvisited = set(range(1, n))

    while unvisited:
        current = path[-1]

        next_node = min(
            unvisited,
            key=lambda i: math.dist(
                corrected_map[current],
                corrected_map[i]
            )
        )
        path.append(next_node)
        unvisited.remove(next_node)
    improved = True

    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n - 1):
                a = corrected_map[path[i]]
                b = corrected_map[path[i + 1]]
                c = corrected_map[path[j]]
                d = corrected_map[path[j + 1]]
                if segments_cross(a, b, c, d):
                    path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                    improved = True
                    break
            if improved:
                break
    return path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # Step 1 -- Generate the map (provided) ---------------------------------
    map_points = generate_map(
        num_points=NUM_POINTS,
        x_range=X_RANGE,
        y_range=Y_RANGE,
        seed=MAP_SEED,
    )

    # Step 2 -- Generate the observations (provided) ------------------------
    observations = generate_observations(
        map_points,
        noise_std=NOISE_STD,
        observation_rate=OBSERVATION_RATE,
        seed=OBSERVATIONS_SEED,
    )
    visualize_map_n_observations(map_points, observations, X_RANGE, Y_RANGE)

    # Step 3 -- Data association (yours) -------------------------------------
    try:
        associations = associate(map_points, observations)
    except NotImplementedError:
        print("Step 3 is not implemented yet: fill in associate() in solution.py.")
        return

    # Step 4 -- Map correction (yours) ---------------------------------------
    try:
        corrected_map = correct_map(map_points, observations, associations)
    except NotImplementedError:
        print("Step 4 is not implemented yet: fill in correct_map() in solution.py.")
        return
    visualize_association_n_correction(
        map_points,
        observations,
        corrected_map,
        associations,
        X_RANGE,
        Y_RANGE,
    )

    # Step 5 -- Path finding (yours) -----------------------------------------
    try:
        path = find_path(corrected_map)
    except NotImplementedError:
        print("Step 5 is not implemented yet: fill in find_path() in solution.py.")
        return
    visualize_path(corrected_map, path)

    print(
        f"Pipeline complete: {len(associations)} associations over "
        f"{len(observations)} observations, path over {len(corrected_map)} "
        f"nodes with total length {path_length(corrected_map, path):.2f}."
    )


if __name__ == "__main__":
    main()
